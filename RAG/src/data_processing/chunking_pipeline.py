"""
chunking_pipeline.py – Smart chunking for HUST Sổ tay RAG documents.

Strategy:
1. SECTION-AWARE BODY CHUNKING: Split body theo heading/section, merge tiny, split large.
2. RICH METADATA PREFIX: prefix Bài viết/Chuyên mục/Liên hệ/Mục.
3. FORM-AWARE ATTACHMENT CHUNKING: biểu mẫu / pháp quy / fallback.
4. SUMMARY/ROUTING CHUNKS: 1 summary chunk per bài + per attachment (embed + Qdrant).
5. MARKDOWN TABLE GUARD: bảo vệ bảng khỏi bị cắt giữa hàng.
6. STABLE CHUNK IDs (Mục 6):
   - Body:       {doc_id}_body_{section_slug}_p{page:02d}
   - Attachment: {doc_id}_att_{att_id8}_p{page:02d}
   - Summary doc:{doc_id}_summary
   - Summary att:{doc_id}_att_{att_id8}_sum
7. SEPARATE ATTACHMENT INDICES (Mục 7):
   - attachment_index: thứ tự file trong bài viết
   - attachment_chunk_index: thứ tự chunk trong file đó
8. RELATED LINKS EXTRACTION (Mục 8):
   - Tách dòng "Liên kết trong bài" khỏi body text → lưu vào metadata.related_links
   - Filter attachment vô nghĩa (tên: "tại đây", "TẠI ĐÂY.", ...) khỏi pipeline

Output:
  data/chunks.jsonl
"""

import json
import re
import sys
import io
import hashlib
import unicodedata
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# LangChain import
# ---------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# ── Body chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE        = 1200   # target chars per chunk (range 900-1400)
CHUNK_OVERLAP     = 150    # overlap chars between body chunks (range 120-180)

# ── Section merge / split thresholds ──────────────────────────────────────
MIN_SECTION_CHARS = 350    # merge sections smaller than this
MAX_SECTION_CHARS = 1400   # split sections larger than this

# ── Attachment – biểu mẫu đơn từ ──────────────────────────────────────────
ATT_CHUNK_SIZE    = 1300   # range 1000-1600
ATT_CHUNK_OVERLAP = 100    # low overlap for forms (range 80-120)
FORM_MAX_INTACT   = 1800   # keep entire form as 1 chunk if <= this
MIN_FORM_BLOCK    = 500    # merge tiny form blocks (tiêu ngữ header)

# ── Attachment – văn bản pháp quy ─────────────────────────────────────────
LEGAL_CHUNK_SIZE    = 1500   # range 1200-1800 (cần context dài hơn)
LEGAL_CHUNK_OVERLAP = 180    # range 150-220

# ── Table guard ────────────────────────────────────────────────────────────
MAX_TABLE_CHARS   = 1_800

# ── I/O ───────────────────────────────────────────────────────────────────
INPUT_FILE  = Path('data/rag_documents.jsonl')
OUTPUT_FILE = Path('data/chunks.jsonl')

# Vietnamese-aware separator priority
SEPARATORS = ['\n---\n', '\n\n', '\n', '\u3002', '. ', ', ', ' ']

# Characters that can NEVER start a Vietnamese syllable (mid-word cut signal)
_VIET_MID_VOWELS = set('ăâêôơư')

# ---------------------------------------------------------------------------
# Section boundary regex (body chunking)
# ---------------------------------------------------------------------------
_SECTION_NUMBERED_RE = re.compile(r'^\d{2,3}\.\s')
_SECTION_TAG_RE      = re.compile(r'^\[[\w\s\u00C0-\u024F\u1EA0-\u1EF9]+\]\s*$')

# ---------------------------------------------------------------------------
# Văn bản pháp quy regex
# ---------------------------------------------------------------------------
_LEGAL_DIEU_RE   = re.compile(r'(?m)^Điều\s+\d+[\.\:]?\s')
_LEGAL_CHUONG_RE = re.compile(r'(?m)^Chương\s+[IVXLCDM\d]+[\.\:]?\s')
_LEGAL_MUC_RE    = re.compile(r'(?m)^Mục\s+\d+[\.\:]?\s')
# Khoản cấp dưới: "1. " hoặc "a) " ở đầu dòng
_LEGAL_KHOAN_RE  = re.compile(r'(?m)^\d+\.\s|^[a-z]\)\s')

# ---------------------------------------------------------------------------
# Form-document signature patterns
# ---------------------------------------------------------------------------
_FORM_SIGNATURE_RE = re.compile(
    r'(CỘNG HÒA|CỘNG HOÀ|Độc lập|Kính gửi|Họ (và |)tên|MSSV|Tôi xin cam|BỘ GIÁO DỤC)',
    re.IGNORECASE,
)

# Form block boundary patterns
_FORM_BLOCK_BOUNDARIES = re.compile(
    r'(?m)^('
    r'CỘNG HÒA|CỘNG HOÀ|BỘ GIÁO DỤC'
    r'|Họ (và |)tên\b|MSSV\b|Lớp\b|Khóa\b'
    r'|Kính đề nghị|Tôi (kính |)đề nghị|Đề nghị\b|Tôi xin\b'
    r'|Tôi xin cam|Cam kết\b|Chữ ký\b'
    r'|Lưu ý\b|Hồ sơ kèm theo'
    r')',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# LangChain Splitters
# ---------------------------------------------------------------------------
_body_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator=True,
    is_separator_regex=False,
    length_function=len,
    add_start_index=False,
)

_att_splitter = RecursiveCharacterTextSplitter(
    chunk_size=ATT_CHUNK_SIZE,
    chunk_overlap=ATT_CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator=True,
    is_separator_regex=False,
    length_function=len,
    add_start_index=False,
)

_legal_splitter = RecursiveCharacterTextSplitter(
    chunk_size=LEGAL_CHUNK_SIZE,
    chunk_overlap=LEGAL_CHUNK_OVERLAP,
    separators=['\n\n', '\n', '. ', ', ', ' '],
    keep_separator=True,
    is_separator_regex=False,
    length_function=len,
    add_start_index=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_emojis_and_symbols(value: str) -> str:
    if not value:
        return ""
    cleaned = []
    for ch in value:
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in ("\n", "\r", "\t"):
            continue
        if cat == "So" or ch in ("□", "☐", "■", "▪", "▫", "♦", "●", "○",
                                  "★", "☆", "▶", "►", "◄", "▼", "▲"):
            continue
        cleaned.append(ch)
    res = "".join(cleaned)
    res = re.sub(r' +', ' ', res)
    return res.strip()


# ---------------------------------------------------------------------------
# Markdown Table Guard
# ---------------------------------------------------------------------------

_TABLE_BLOCK_RE = re.compile(r'(?m)^(\|.+\|\s*\n)+')


def _protect_tables(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    counter = [0]

    def _replace_table(m: re.Match) -> str:
        block = m.group(0).rstrip('\n')
        if len(block) <= MAX_TABLE_CHARS:
            token = f'\x00TABLE{counter[0]:04d}\x00'
            counter[0] += 1
            placeholders[token] = block
            return f'\n{token}\n'
        else:
            rows = block.split('\n')
            groups: list[str] = []
            current_rows: list[str] = []
            current_len = 0
            for row in rows:
                row_len = len(row) + 1
                if current_rows and current_len + row_len > MAX_TABLE_CHARS:
                    groups.append('\n'.join(current_rows))
                    current_rows = [row]
                    current_len = row_len
                else:
                    current_rows.append(row)
                    current_len += row_len
            if current_rows:
                groups.append('\n'.join(current_rows))
            parts = []
            for group in groups:
                token = f'\x00TABLE{counter[0]:04d}\x00'
                counter[0] += 1
                placeholders[token] = group
                parts.append(token)
            return '\n\n' + '\n\n'.join(parts) + '\n\n'

    patched = _TABLE_BLOCK_RE.sub(_replace_table, text)
    return patched, placeholders


def _restore_tables(chunks: list[str], placeholders: dict[str, str]) -> list[str]:
    if not placeholders:
        return chunks
    restored = []
    for chunk in chunks:
        for token, original in placeholders.items():
            chunk = chunk.replace(token, original)
        restored.append(chunk)
    return restored


# ---------------------------------------------------------------------------
# Stable ID helpers  (Mục 6)
# ---------------------------------------------------------------------------

# Vietnamese diacritic → ASCII table for slug generation
_VI_TRANS = str.maketrans(
    'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ',
    'aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd'
    'AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD',
)


def _make_section_slug(text: str, max_len: int = 24) -> str:
    """
    Tạo slug ổn định từ section header hoặc tên attachment.
    Ví dụ: "31. Hoãn thi cuối kỳ" → "31_hoan_thi_cuoi_ky"
    """
    if not text:
        return 'sec'
    translated = text.translate(_VI_TRANS)
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', translated).lower().strip('_')
    return slug[:max_len] or 'sec'


# ---------------------------------------------------------------------------
# Junk attachment filter  (Mục 8)
# ---------------------------------------------------------------------------

# Tên attachment vô nghĩa — chỉ là anchor text, không phải tên file thực
_JUNK_ATT_NAMES = frozenset({
    'tại đây', 'tại đây.', 'tai day', 'tai day.',
    'ở đây', 'ở đây.', 'o day', 'o day.',
    'xem tại đây', 'xem tại đây.',
    'here', 'click here', 'download here',
    'tại đây!', 'xem thêm',
})


def _is_junk_attachment(att: dict) -> bool:
    """
    Phát hiện attachment là anchor text vô nghĩa ("tại đây", "TẠI ĐÂY.", ...).
    Những file này không có tên có ý nghĩa và không cần embed.
    """
    name = (att.get('name') or '').strip()
    if len(name) < 5:
        return True
    if name.lower() in _JUNK_ATT_NAMES:
        return True
    # Nếu tên chỉ toàn IN HOA và dưới 12 ký tự → thường là anchor
    if name == name.upper() and len(name) <= 12 and not any(c.isdigit() for c in name):
        return True
    return False


# ---------------------------------------------------------------------------
# Related-links extractor  (Mục 8)
# ---------------------------------------------------------------------------

# Dòng "- Liên kết trong bài '...': anchor_text" do crawler tổng hợp
_RELATED_LINK_LINE_RE = re.compile(
    r'^-\s*Liên kết trong bài [\'"].+?[\'"]:.+$',
    re.MULTILINE,
)
# Header "Liên kết liên quan:" (standalone line)
_RELATED_LINK_HEADER_RE = re.compile(
    r'Liên kết liên quan:\s*\n',
)


def _extract_related_links(text: str) -> tuple[str, list[str]]:
    """
    Tách các dòng inline-link thêm vào bởi crawler ra khỏi body.
    Những dòng này dạng:
      "Liên kết liên quan:\n- Liên kết trong bài '...': anchor"

    Returns:
        (text_without_links, list_of_removed_link_lines)

    Lưu ý: Chỉ xoá các dòng link meta, KHÔNG xoá nội dung bài sau dòng link.
    """
    links_found = _RELATED_LINK_LINE_RE.findall(text)
    cleaned = _RELATED_LINK_LINE_RE.sub('', text)
    # Xoá header nếu sau header không còn nội dung link (chỉ trống)
    cleaned = _RELATED_LINK_HEADER_RE.sub('', cleaned)
    # Gộp nhiều dòng trống thành tối đa 2
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip(), links_found



# ---------------------------------------------------------------------------
# Post-split validation & merging
# ---------------------------------------------------------------------------

def _starts_with_hard_cut(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in _VIET_MID_VOWELS


def _merge_hard_cuts(parts: list[str], sep: str = ' ') -> list[str]:
    if not parts:
        return parts
    merged: list[str] = [parts[0]]
    for part in parts[1:]:
        if _starts_with_hard_cut(part):
            merged[-1] = (merged[-1].rstrip() + sep + part.lstrip()).strip()
        else:
            merged.append(part)
    return merged


# ---------------------------------------------------------------------------
# Prefix builder
# ---------------------------------------------------------------------------

def _build_prefix(
    title: str,
    category: str,
    emails: list[str],
    section_header: str = '',
    is_attachment: bool = False,
    att_name: str = '',
) -> str:
    lines = [
        f"Bài viết: {title}",
        f"Chuyên mục: {category}",
    ]
    if is_attachment:
        lines.append(f"Biểu mẫu/Tài liệu đính kèm: {att_name}")
    elif section_header:
        lines.append(f"Mục: {section_header}")
    if emails:
        lines.append(f"Liên hệ: {', '.join(emails[:2])}")
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Section-aware body chunking
# ---------------------------------------------------------------------------

def _is_section_boundary(line: str) -> bool:
    s = line.strip()
    return bool(_SECTION_NUMBERED_RE.match(s) or _SECTION_TAG_RE.match(s))


def _parse_sections(text: str) -> list[tuple[str, str]]:
    lines = text.split('\n')
    sections: list[tuple[str, str]] = []
    current_header = ''
    current_lines: list[str] = []

    for line in lines:
        if _is_section_boundary(line):
            if current_lines:
                sections.append((current_header, '\n'.join(current_lines)))
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, '\n'.join(current_lines)))

    return sections


def _split_large_section(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    patched, placeholders = _protect_tables(text)
    raw = _body_splitter.split_text(patched)
    raw = _restore_tables(raw, placeholders)
    raw = [c.strip() for c in raw if c.strip()]
    final: list[str] = []
    for chunk in raw:
        if len(chunk) > int(CHUNK_SIZE * 1.5):
            sub = _body_splitter.split_text(chunk)
            final.extend(s.strip() for s in sub if s.strip())
        else:
            final.append(chunk)
    return [c for c in final if c]


def _split_body_by_sections(text: str) -> list[tuple[str, str]]:
    """
    Section-aware body split. Returns list of (section_header, chunk_text).
    Phase 1 – detect, Phase 2 – merge tiny, Phase 3 – split large.
    """
    raw_sections = _parse_sections(text)

    # Phase 2: Merge tiny sections
    merged: list[tuple[str, str]] = []
    buf_header = ''
    buf_content = ''

    for header, content in raw_sections:
        size = len(content.strip())
        if not buf_content:
            buf_header = header
            buf_content = content
        elif size < MIN_SECTION_CHARS:
            buf_content = buf_content.rstrip() + '\n\n' + content.strip()
        else:
            if buf_content.strip():
                merged.append((buf_header, buf_content.strip()))
            buf_header = header
            buf_content = content

    if buf_content.strip():
        merged.append((buf_header, buf_content.strip()))

    # Phase 3: Split large sections
    result: list[tuple[str, str]] = []
    for header, content in merged:
        if len(content) > MAX_SECTION_CHARS:
            sub_chunks = _split_large_section(content)
            sub_chunks = _merge_hard_cuts(sub_chunks)
            for chunk in sub_chunks:
                result.append((header, chunk))
        else:
            result.append((header, content))

    return result


# ---------------------------------------------------------------------------
# Văn bản pháp quy chunking (Strategy 3)
# ---------------------------------------------------------------------------

def _is_legal_doc(text: str) -> bool:
    """
    Phát hiện văn bản pháp quy: có từ khoá Điều hoặc Chương ở đầu dòng.
    Kiểm tra trong 3000 chars đầu để tránh false-positive.
    """
    sample = text[:3000]
    return bool(_LEGAL_DIEU_RE.search(sample) or _LEGAL_CHUONG_RE.search(sample))


def _split_legal_doc(text: str) -> list[str]:
    """
    Split văn bản pháp quy theo cấu trúc Điều/Khoản.

    Ưu tiên: không cắt giữa một Điều.
    Nếu Điều quá dài → split thêm theo Khoản (1., 2., a), b)).
    Size: LEGAL_CHUNK_SIZE (1500), overlap: LEGAL_CHUNK_OVERLAP (180).
    """
    text = text.strip()
    if not text:
        return []

    # Tìm tất cả vị trí bắt đầu của Điều
    boundaries = [m.start() for m in _LEGAL_DIEU_RE.finditer(text)]

    # Nếu không tìm thấy Điều, thử Chương/Mục
    if not boundaries:
        boundaries = [m.start() for m in _LEGAL_CHUONG_RE.finditer(text)]
    if not boundaries:
        boundaries = [m.start() for m in _LEGAL_MUC_RE.finditer(text)]

    # Không có cấu trúc pháp quy rõ → fallback
    if len(boundaries) < 2:
        return _recursive_split_with_table_guard(text, _legal_splitter, LEGAL_CHUNK_SIZE)

    # Tách thành các Điều
    dieu_blocks: list[str] = []
    for i, pos in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        block = text[pos:end].strip()
        if block:
            dieu_blocks.append(block)

    # Phần intro trước Điều đầu tiên
    intro = text[:boundaries[0]].strip()

    # Gộp và split từng Điều
    result: list[str] = []

    if intro and len(intro) >= 50:
        # Intro (phần mở đầu) có thể dài, split nếu cần
        if len(intro) > LEGAL_CHUNK_SIZE:
            result.extend(_recursive_split_with_table_guard(intro, _legal_splitter, LEGAL_CHUNK_SIZE))
        else:
            result.append(intro)

    # Buffer để merge Điều ngắn
    buf = ''
    for block in dieu_blocks:
        combined = (buf + '\n\n' + block).strip() if buf else block
        if len(combined) <= LEGAL_CHUNK_SIZE:
            buf = combined
        else:
            # Flush buffer
            if buf:
                result.append(buf)
            # Block này có thể quá dài → split theo Khoản
            if len(block) > LEGAL_CHUNK_SIZE:
                result.extend(_split_by_khoan(block))
            else:
                buf = block

    if buf:
        result.append(buf)

    result = _merge_hard_cuts(result)
    return [c for c in result if c and len(c) >= 50]


def _split_by_khoan(dieu_text: str) -> list[str]:
    """Split một Điều dài theo Khoản (1., 2., a), b))."""
    boundaries = [m.start() for m in _LEGAL_KHOAN_RE.finditer(dieu_text)]
    if len(boundaries) < 2:
        return _recursive_split_with_table_guard(dieu_text, _legal_splitter, LEGAL_CHUNK_SIZE)

    khoan_blocks: list[str] = []
    header = dieu_text[:boundaries[0]].strip()  # Tên Điều
    for i, pos in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(dieu_text)
        block = dieu_text[pos:end].strip()
        if block:
            khoan_blocks.append(block)

    result: list[str] = []
    buf = header + '\n' if header else ''
    for khoan in khoan_blocks:
        candidate = (buf + '\n' + khoan).strip()
        if len(candidate) <= LEGAL_CHUNK_SIZE:
            buf = candidate
        else:
            if buf:
                result.append(buf)
            buf = (header + '\n' + khoan).strip() if header else khoan

    if buf:
        result.append(buf)

    return [c for c in result if c]


def _recursive_split_with_table_guard(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    size_limit: int,
) -> list[str]:
    """Dùng splitter với table guard. Dùng cho fallback."""
    patched, placeholders = _protect_tables(text)
    raw = splitter.split_text(patched)
    raw = _restore_tables(raw, placeholders)
    raw = [c.strip() for c in raw if c.strip()]
    final: list[str] = []
    for chunk in raw:
        if len(chunk) > int(size_limit * 1.5):
            sub = splitter.split_text(chunk)
            final.extend(s.strip() for s in sub if s.strip())
        else:
            final.append(chunk)
    return [c for c in final if c]


# ---------------------------------------------------------------------------
# Form-aware attachment chunking
# ---------------------------------------------------------------------------

def _split_by_form_blocks(text: str) -> list[str]:
    """Split biểu mẫu đơn từ VN theo khối logic."""
    boundaries = [m.start() for m in _FORM_BLOCK_BOUNDARIES.finditer(text)]
    if len(boundaries) < 2:
        return []

    raw_blocks: list[str] = []
    prev = 0
    for pos in boundaries[1:]:
        block = text[prev:pos].strip()
        if block:
            raw_blocks.append(block)
        prev = pos
    tail = text[prev:].strip()
    if tail:
        raw_blocks.append(tail)

    # Pass 1: merge tiny blocks forward
    merged1: list[str] = []
    pending = ''
    for block in raw_blocks:
        combined = (pending + '\n\n' + block).strip() if pending else block
        if len(combined) < MIN_FORM_BLOCK:
            pending = combined
        else:
            merged1.append(combined)
            pending = ''
    if pending:
        if merged1:
            merged1[-1] = merged1[-1] + '\n\n' + pending
        else:
            merged1.append(pending)

    # Pass 2: merge blocks that fit within ATT_CHUNK_SIZE
    merged2: list[str] = []
    buf = ''
    for block in merged1:
        if not buf:
            buf = block
        elif len(buf) + len(block) + 2 <= ATT_CHUNK_SIZE:
            buf = buf + '\n\n' + block
        else:
            merged2.append(buf)
            buf = block
    if buf:
        merged2.append(buf)

    # Pass 3: split any block still oversized
    final: list[str] = []
    for block in merged2:
        if len(block) > int(ATT_CHUNK_SIZE * 1.5):
            sub = _att_splitter.split_text(block)
            final.extend(s.strip() for s in sub if s.strip())
        else:
            final.append(block)

    return final if len(final) > 1 else []


def _split_attachment(text: str) -> list[str]:
    """
    Smart attachment chunking — routes to the correct strategy:
    1. Ngắn (≤ FORM_MAX_INTACT)      → 1 chunk nguyên
    2. Văn bản pháp quy               → split theo Điều/Khoản
    3. Biểu mẫu đơn từ                → split theo khối logic form
    4. Fallback                       → RecursiveCharacterTextSplitter
    """
    text = text.strip()
    if not text:
        return []

    # Rule 1: Short → keep intact
    if len(text) <= FORM_MAX_INTACT:
        return [text]

    # Rule 2: Văn bản pháp quy (quy chế, quy định)
    if _is_legal_doc(text):
        return _split_legal_doc(text)

    # Rule 3: Biểu mẫu đơn từ
    if _FORM_SIGNATURE_RE.search(text[:2000]):
        blocks = _split_by_form_blocks(text)
        if blocks:
            final = _merge_hard_cuts(blocks)
            return [c for c in final if c]

    # Rule 4: Fallback
    return _recursive_split_with_table_guard(text, _att_splitter, ATT_CHUNK_SIZE)


# ---------------------------------------------------------------------------
# Summary / routing chunk builder (Strategy 4)
# ---------------------------------------------------------------------------

def _build_doc_summary(doc: dict) -> str | None:
    """
    Tạo summary chunk ngắn (~100-250 chars) cho bài viết.
    Dùng để vector search định tuyến nhanh đến đúng bài.

    Ví dụ:
      "Bài viết '[Ban Đào tạo] Hướng dẫn...' thuộc Học tập.
       Gồm các chủ đề: rút học phần, miễn ngoại ngữ, hoãn thi."
    """
    title = doc.get('title', '')
    cat   = doc['metadata'].get('category_name', '')
    tags  = doc['metadata'].get('tags', [])
    emails = doc['metadata'].get('emails', [])

    if not title:
        return None

    lines = [f"Bài viết '{title}' thuộc {cat}."]
    if tags:
        tag_str = ', '.join(str(t) for t in tags[:6] if str(t).strip())
        if tag_str:
            lines.append(f"Gồm các chủ đề: {tag_str}.")
    if emails:
        lines.append(f"Liên hệ: {', '.join(emails[:2])}.")

    return '\n'.join(lines)


def _build_att_summary(doc: dict, att: dict) -> str | None:
    """
    Tạo summary chunk cho một attachment.
    Dùng để user hỏi "mẫu đơn hoãn thi" → hit đúng attachment.

    Ví dụ:
      "File đính kèm: Đơn hoãn thi. Bài viết: [Ban Đào tạo]...
       Chuyên mục: Học tập. Liên hệ: ..."
    """
    att_name = att.get('name', '')
    title    = doc.get('title', '')
    cat      = doc['metadata'].get('category_name', '')
    emails   = doc['metadata'].get('emails', [])
    tags     = doc['metadata'].get('tags', [])

    if not att_name or not att.get('content', ''):
        return None

    lines = [
        f"File đính kèm: {att_name}.",
        f"Bài viết: {title}. Chuyên mục: {cat}.",
    ]
    if tags:
        tag_str = ', '.join(str(t) for t in tags[:3] if str(t).strip())
        if tag_str:
            lines.append(f"Liên quan: {tag_str}.")
    if emails:
        lines.append(f"Liên hệ: {', '.join(emails[:2])}.")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Build chunks for one document
# ---------------------------------------------------------------------------

def build_chunks(doc: dict) -> list[dict]:
    """
    Build RAG chunks for a single document.

    Chunk types:
      - 'body'         : content chunk từ body bài viết
      - 'attachment'   : content chunk từ file đính kèm
      - 'summary'      : routing chunk – embed + Qdrant, bổ sung cho content
      - 'link_summary' : danh sách liên kết liên quan – lưu metadata, KHÔNG dùng trong LLM context

    IDs ổn định (Mục 6):
      - Body   : {doc_id}_body_{section_slug}_p{page:02d}
      - Att    : {doc_id}_att_{att_id8}_p{page:02d}
      - Sum doc: {doc_id}_summary
      - Sum att: {doc_id}_att_{att_id8}_sum
    """
    base_meta   = dict(doc['metadata'])
    doc_id      = doc['id']
    title       = doc['title']
    category    = doc['metadata'].get('category_name', '')
    emails      = doc['metadata'].get('emails', [])

    raw_text    = doc.get('text', '')
    attachments = doc.get('attachments', [])
    chunks_out: list[dict] = []
    chunk_idx = 0   # global counter trong doc – dùng cho ±1 window lookup

    def _make_chunk(
        text: str,
        chunk_type: str,
        stable_id: str,
        section_header: str = '',
        is_attachment: bool = False,
        att_name: str | None = None,
        extra_meta: dict | None = None,
    ) -> dict:
        nonlocal chunk_idx
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        meta = {
            **base_meta,
            'chunk_index':    chunk_idx,
            'chunk_type':     chunk_type,
            'is_attachment':  is_attachment,
            'section_header': section_header,
            'content_hash':   content_hash,
        }
        if extra_meta:
            meta.update(extra_meta)
        obj = {
            'id':              stable_id,
            'parent_id':       doc_id,
            'chunk_index':     chunk_idx,
            'chunk_type':      chunk_type,
            'is_attachment':   is_attachment,
            'attachment_name': att_name,
            'section_header':  section_header,
            'text':            text,
            'metadata':        meta,
        }
        chunk_idx += 1
        return obj

    # ── Bước 0: tách related links khỏi body (Mục 8) ─────────────────────
    body_text, related_links = _extract_related_links(raw_text)
    if related_links:
        base_meta['related_links'] = related_links   # lưu vào metadata, không embed riêng

    # ── Summary chunk cho bài viết ────────────────────────────────────────
    doc_summary = _build_doc_summary(doc)
    if doc_summary:
        prefix = _build_prefix(title, category, emails)
        chunks_out.append(_make_chunk(
            text         = prefix + doc_summary,
            chunk_type   = 'summary',
            stable_id    = f'{doc_id}_summary',
            is_attachment= False,
        ))

    # ── Body chunks (section-aware) ───────────────────────────────────────
    # Đếm riêng page per section để tạo stable ID
    section_page_counter: dict[str, int] = {}   # slug → page_count

    section_chunks = _split_body_by_sections(body_text)
    for section_header, content in section_chunks:
        content = content.strip()
        if not content:
            continue

        # Stable ID: dùng section_slug + page index trong section đó
        sec_slug = _make_section_slug(section_header) if section_header else 'body'
        page     = section_page_counter.get(sec_slug, 0)
        section_page_counter[sec_slug] = page + 1
        stable_id = f'{doc_id}_body_{sec_slug}_p{page:02d}'

        prefix    = _build_prefix(title, category, emails, section_header=section_header)
        full_text = prefix + content
        chunks_out.append(_make_chunk(
            text           = full_text,
            chunk_type     = 'body',
            stable_id      = stable_id,
            section_header = section_header,
            is_attachment  = False,
        ))

    # ── Attachment chunks (form/legal/fallback) + summary ─────────────────
    for att_idx, att in enumerate(attachments):   # att_idx: thứ tự attachment trong bài
        # Lọc attachment vô nghĩa (Mục 8)
        if _is_junk_attachment(att):
            continue

        att_name    = att.get('name', 'Tài liệu đính kèm')
        att_content = att.get('content', '')
        # Stable att key: att_id_raw + att_idx để đảm bảo unique
        # kể cả khi cùng 1 file được link 2 lần trong bài
        att_id_raw  = (att.get('attachment_id') or 'unknown00')[:8]
        att_id8     = f'{att_id_raw}_{att_idx:02d}'
        if not att_content:
            continue

        # Summary chunk cho attachment (Mục 6 stable ID)
        att_summary_text = _build_att_summary(doc, att)
        if att_summary_text:
            att_prefix = _build_prefix(
                title, category, emails,
                is_attachment=True, att_name=att_name,
            )
            chunks_out.append(_make_chunk(
                text          = att_prefix + att_summary_text,
                chunk_type    = 'summary',
                stable_id     = f'{doc_id}_att_{att_id8}_sum',
                is_attachment = True,
                att_name      = att_name,
                extra_meta    = {
                    'attachment_index':      att_idx,    # Mục 7: thứ tự attachment trong bài
                    'attachment_chunk_index': -1,        # -1 = summary, không phải content chunk
                    'attachment_name': att_name,
                    'attachment_url':  att.get('url'),
                    'fetch_status':    att.get('fetch_status'),
                },
            ))

        # Content chunks (Mục 7: đếm riêng per-attachment)
        att_content  = att_content[:60_000]
        att_parts    = _split_attachment(att_content)
        att_chunk_p  = 0   # page index trong attachment này – reset mỗi attachment

        for part in att_parts:
            part = part.strip()
            if not part or len(part) < 50:
                continue

            # Stable ID: att_{att_id8}_p{page} (Mục 6)
            stable_id = f'{doc_id}_att_{att_id8}_p{att_chunk_p:02d}'

            prefix    = _build_prefix(
                title, category, emails,
                is_attachment=True, att_name=att_name,
            )
            full_text = prefix + part
            chunks_out.append(_make_chunk(
                text          = full_text,
                chunk_type    = 'attachment',
                stable_id     = stable_id,
                is_attachment = True,
                att_name      = att_name,
                extra_meta    = {
                    'attachment_index':       att_idx,       # Mục 7: thứ tự att trong bài
                    'attachment_chunk_index': att_chunk_p,   # Mục 7: chunk thứ mấy trong att
                    'attachment_id':          att.get('attachment_id'),
                    'attachment_name':        att_name,
                    'attachment_url':         att.get('url'),
                    'local_filename':         att.get('local_filename'),
                    'source_link_index':      att.get('link_index'),
                    'fetch_status':           att.get('fetch_status'),
                },
            ))
            att_chunk_p += 1

    return chunks_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_chunks: list[dict] = []
    docs_processed: int = 0
    junk_atts_filtered: int = 0
    docs_with_related_links: int = 0
    related_links_total: int = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc    = json.loads(line)

            # Count junk attachments before chunking
            junk_count = sum(1 for a in doc.get('attachments', []) if _is_junk_attachment(a))
            junk_atts_filtered += junk_count

            # Count related links
            _, rl = _extract_related_links(doc.get('text', ''))
            if rl:
                docs_with_related_links += 1
                related_links_total += len(rl)

            chunks = build_chunks(doc)
            all_chunks.extend(chunks)
            docs_processed += 1

    # Count remaining hard-cuts (should be 0)
    hard_cut_fixed = 0
    for chunk in all_chunks:
        body_text = '\n'.join(chunk['text'].split('\n')[4:])
        if body_text and body_text.lstrip() and body_text.lstrip()[0] in _VIET_MID_VOWELS:
            hard_cut_fixed += 1

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='\n') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Statistics
    lengths      = [len(c['text']) for c in all_chunks]
    body_chunks  = [c for c in all_chunks if c['chunk_type'] == 'body']
    att_chunks   = [c for c in all_chunks if c['chunk_type'] == 'attachment']
    sum_chunks   = [c for c in all_chunks if c['chunk_type'] == 'summary']
    over_2000    = [c for c in all_chunks if len(c['text']) > 2000]
    tiny         = [c for c in all_chunks if len(c['text']) < 200]

    # Check ID stability: verify no duplicate IDs
    all_ids = [c['id'] for c in all_chunks]
    dup_ids = len(all_ids) - len(set(all_ids))

    print(f"\n✅ Chunking hoàn tất! (Section-Aware + Form-Aware + Legal + Summary + Stable IDs)")
    print(f"   Tổng docs đầu vào    : {docs_processed}")
    print(f"   Tổng chunks đầu ra   : {len(all_chunks)}")
    print(f"     - Body chunks      : {len(body_chunks)}")
    print(f"     - Attach chunks    : {len(att_chunks)}")
    print(f"     - Summary chunks   : {len(sum_chunks)}")
    print(f"   Mục 6 - Stable IDs:")
    print(f"     - Duplicate IDs    : {dup_ids}  (phải = 0)")
    print(f"     - Sample ID: {all_chunks[1]['id'] if len(all_chunks) > 1 else 'N/A'}")
    print(f"   Mục 7 - Attachment indices:")
    att_with_idx = [c for c in att_chunks if 'attachment_chunk_index' in c.get('metadata', {})]
    print(f"     - Att chunks có attachment_chunk_index: {len(att_with_idx)}/{len(att_chunks)}")
    print(f"   Mục 8 - Related links & Junk filter:")
    print(f"     - Junk attachments loại bỏ  : {junk_atts_filtered}")
    print(f"     - Docs có related links   : {docs_with_related_links}/{docs_processed}")
    print(f"     - Tổng dòng link đã tách  : {related_links_total}")
    print(f"   Độ dài chunk:")
    if lengths:
        print(f"     - Trung bình       : {sum(lengths)/len(lengths):.0f} chars")
        print(f"     - Lớn nhất         : {max(lengths)} chars")
        print(f"     - Nhỏ nhất         : {min(lengths)} chars")
    print(f"   Kiểm tra chất lượng:")
    print(f"     - Chunks vượt 2000 chars    : {len(over_2000)}")
    print(f"     - Chunks rất ngắn <200      : {len(tiny)}")
    print(f"     - Hard-cut còn sót (ideal=0): {hard_cut_fixed}")
    if over_2000:
        print("   ⚠  Chunks > 2000 chars (top 5):")
        for c in over_2000[:5]:
            print(f"       {c['id']} – {len(c['text'])} chars [{c['chunk_type']}]")
    if dup_ids > 0:
        print("   ❌ DUPLICATE IDs detected! Cần kiểm tra lại slug generation:")
        from collections import Counter
        id_counts = Counter(all_ids)
        for cid, cnt in id_counts.most_common(5):
            if cnt > 1:
                print(f"       {cid}: {cnt} lần")

    # Sample stable IDs
    print("\n   📄 Sample Stable IDs:")
    shown = set()
    for c in all_chunks:
        ctype = c['chunk_type']
        if ctype not in shown and len(shown) < 4:
            print(f"     [{ctype:10s}] {c['id']}")
            shown.add(ctype)

    # Sample summary chunks
    print("\n   📋 Sample SUMMARY chunks:")
    shown_sum = 0
    for c in all_chunks:
        if c['chunk_type'] == 'summary' and shown_sum < 2:
            print(f"\n   [SUMMARY {'ATT' if c['is_attachment'] else 'DOC'}] {c['id']}")
            for ln in c['text'].split('\n')[:4]:
                print(f"     {ln}")
            shown_sum += 1

    print(f"\n   📄 Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
