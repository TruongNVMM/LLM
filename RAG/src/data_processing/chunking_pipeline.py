#!/usr/bin/env python3
"""
chunking_pipeline.py – Smart chunking for HUST Sổ tay RAG documents.

Strategy (LangChain-powered):
1. MARKDOWN TABLE GUARD: Detect Markdown tables inside text; keep small tables
   intact (≤ MAX_TABLE_CHARS) so they are never split mid-row.
2. BODY CHUNKING: Use LangChain RecursiveCharacterTextSplitter on the article
   body with Vietnamese-aware separators:
       \\n\\n  >  \\n  >  。  >  .  >  ,  >  space
   LangChain guarantees word-boundary splits – no more hard-cuts mid-word.
3. ATTACHMENT CHUNKING: Same splitter, slightly larger size to keep form
   structures coherent; each attachment block becomes independent chunk(s).
4. CONTEXT PRESERVATION: Every chunk carries:
   - A 1-2 line prefix: "Bài viết: <title> | Chuyên mục: <category>"
   - Full metadata inherited from parent doc
   - chunk_index, chunk_type (body | attachment), is_attachment flag
5. POST-VALIDATION: After splitting, verify no chunk starts with a Vietnamese
   vowel-only grapheme (ă â ê ô ơ ư) which signals a hard-cut mid-word.
   Any such chunk is merged back into the previous chunk.

Output:
  data/processed/rag_chunks.jsonl
"""

import json
import re
import sys
import io
import hashlib
import unicodedata
from pathlib import Path
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# LangChain import
# ---------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE             = 1500   # target chars per chunk (≈ 300-400 tokens)
CHUNK_OVERLAP          = 200    # overlap chars between consecutive body chunks
ATT_CHUNK_SIZE         = 1500   # attachment chunk size (same, for consistency)
ATT_CHUNK_OVERLAP      = 100    # attachment needs less overlap
MAX_ATTACHMENT_CHARS   = 10_000 # hard cap per attachment before splitting
MAX_TABLE_CHARS        = 1_800  # tables smaller than this are kept intact
INPUT_FILE  = Path('data/processed/rag_documents.jsonl')
OUTPUT_FILE = Path('data/processed/rag_chunks.jsonl')

# Vietnamese-aware separator priority (\n---\n handles Markdown HR; \n\n handles paragraphs)
SEPARATORS = ['\n---\n', '\n\n', '\n', '\u3002', '. ', ', ', ' ']

# Characters that can NEVER start a Vietnamese syllable (mid-word hard-cut signal)
_VIET_MID_VOWELS = set('ăâêôơư')

# ---------------------------------------------------------------------------
# LangChain Splitter instances (shared, stateless)
# ---------------------------------------------------------------------------
_body_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator=True,          # keep punctuation/newline at chunk boundary
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_emojis_and_symbols(value: str) -> str:
    """Strip control characters and problematic Unicode symbols."""
    if not value:
        return ""
    cleaned = []
    for ch in value:
        cat = unicodedata.category(ch)
        # Strip control characters (Cc) except newline, carriage return, and tab
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

_TABLE_BLOCK_RE = re.compile(
    r'(?m)^(\|.+\|\s*\n)+',   # one or more consecutive table rows
)


def _protect_tables(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace Markdown table blocks with placeholder tokens so the splitter
    never cuts them in half.  Returns (patched_text, {token: original_block}).

    - Small tables (≤ MAX_TABLE_CHARS) → single atomic token.
    - Large tables → split into ROW GROUPS of ~MAX_TABLE_CHARS each so that
      the splitter treats each group as an independent paragraph, but rows
      within a group are never torn apart.
    """
    placeholders: dict[str, str] = {}
    counter = [0]

    def _replace_table(m: re.Match) -> str:
        block = m.group(0).rstrip('\n')
        if len(block) <= MAX_TABLE_CHARS:
            # Keep whole table as one atomic unit
            token = f'\x00TABLE{counter[0]:04d}\x00'
            counter[0] += 1
            placeholders[token] = block
            return f'\n{token}\n'
        else:
            # Group rows until the group reaches MAX_TABLE_CHARS,
            # then start a new group (each group becomes one token).
            rows   = block.split('\n')
            groups: list[str] = []
            current_rows: list[str] = []
            current_len  = 0
            for row in rows:
                row_len = len(row) + 1  # +1 for the '\n'
                if current_rows and current_len + row_len > MAX_TABLE_CHARS:
                    groups.append('\n'.join(current_rows))
                    current_rows = [row]
                    current_len  = row_len
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
    """Restore table tokens back to their original content."""
    if not placeholders:
        return chunks
    restored = []
    for chunk in chunks:
        for token, original in placeholders.items():
            chunk = chunk.replace(token, original)
        restored.append(chunk)
    return restored


# ---------------------------------------------------------------------------
# Post-split validation & merging
# ---------------------------------------------------------------------------

def _starts_with_hard_cut(text: str) -> bool:
    """
    Return True if the chunk starts with a character that can never begin
    a Vietnamese syllable (ă â ê ô ơ ư), indicating a mid-word hard-cut.
    """
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in _VIET_MID_VOWELS


def _merge_hard_cuts(parts: list[str], sep: str = ' ') -> list[str]:
    """
    Merge any part that starts with a hard-cut character into the previous part.
    This is a safety net – LangChain should not produce these, but we guard
    against degenerate inputs (e.g., very long URLs with no spaces).
    """
    if not parts:
        return parts
    merged: list[str] = [parts[0]]
    for part in parts[1:]:
        if _starts_with_hard_cut(part):
            # Merge into previous
            merged[-1] = (merged[-1].rstrip() + sep + part.lstrip()).strip()
        else:
            merged.append(part)
    return merged


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _split_body(text: str) -> list[str]:
    """Split body text using LangChain, with table protection."""
    text = text.strip()
    if not text:
        return []

    patched, placeholders = _protect_tables(text)
    raw_chunks = _body_splitter.split_text(patched)
    raw_chunks = _restore_tables(raw_chunks, placeholders)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    # Safety second-pass: split any chunk that is still over 1.5× CHUNK_SIZE
    # (can happen when a restored table-group itself is very large)
    final: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) > int(CHUNK_SIZE * 1.5):
            sub = _body_splitter.split_text(chunk)
            final.extend(s.strip() for s in sub if s.strip())
        else:
            final.append(chunk)
    final = _merge_hard_cuts(final)
    return [c for c in final if c]


def _split_attachment(text: str) -> list[str]:
    """Split attachment text using LangChain, with table protection."""
    text = text.strip()
    if not text:
        return []

    patched, placeholders = _protect_tables(text)
    raw_chunks = _att_splitter.split_text(patched)
    raw_chunks = _restore_tables(raw_chunks, placeholders)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    # Safety second-pass for oversized chunks
    final: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) > int(ATT_CHUNK_SIZE * 1.5):
            sub = _att_splitter.split_text(chunk)
            final.extend(s.strip() for s in sub if s.strip())
        else:
            final.append(chunk)
    final = _merge_hard_cuts(final)
    return [c for c in final if c]


# ---------------------------------------------------------------------------
# Build chunks for one document
# ---------------------------------------------------------------------------

def build_chunks(doc: dict) -> list[dict]:
    """
    Build RAG chunks for a single document.

    Each chunk has:
      id, parent_id, chunk_index, chunk_type, is_attachment,
      attachment_name, text (prefix + content), metadata
    """
    base_meta  = dict(doc['metadata'])
    title      = doc['title']
    cat        = doc['metadata'].get('category_name', '')
    prefix     = f"Bài viết: {title} | Chuyên mục: {cat}\n"

    text                = doc.get('text', '')
    attachments         = doc.get('attachments', [])
    chunks_out: list[dict] = []
    chunk_idx           = 0

    # ── Body chunks ──────────────────────────────────────────────────────────
    body_parts = _split_body(text)
    for part in body_parts:
        if not part.strip():
            continue
            
        full_text = prefix + part
        content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
        
        chunks_out.append({
            'id':              f"{doc['id']}_chunk{chunk_idx:03d}",
            'parent_id':       doc['id'],
            'chunk_index':     chunk_idx,
            'chunk_type':      'body',
            'is_attachment':   False,
            'attachment_name': None,
            'text':            full_text,
            'metadata': {
                **base_meta,
                'chunk_index':  chunk_idx,
                'chunk_type':   'body',
                'is_attachment': False,
                'content_hash': content_hash
            },
        })
        chunk_idx += 1

    # ── Attachment chunks ────────────────────────────────────────────────────
    for att_chunk_idx, att in enumerate(attachments):
        att_name = att.get('name', 'Tài liệu đính kèm')
        att_content = att.get('content', '')
        if not att_content:
            continue
            
        att_content = att_content[:MAX_ATTACHMENT_CHARS]   # hard cap
        att_parts   = _split_attachment(att_content)
        att_prefix  = (
            f"Bài viết: {title} | Chuyên mục: {cat}\n"
            f"Biểu mẫu/Tài liệu đính kèm: {att_name}\n"
        )
        for part in att_parts:
            if not part.strip():
                continue
                
            full_text = att_prefix + part
            content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
            
            chunks_out.append({
                'id':              f"{doc['id']}_chunk{chunk_idx:03d}",
                'parent_id':       doc['id'],
                'chunk_index':     chunk_idx,
                'chunk_type':      'attachment',
                'is_attachment':   True,
                'attachment_name': att_name,
                'text':            full_text,
                'metadata': {
                    **base_meta,
                    'chunk_type':    'attachment',
                    'is_attachment': True,
                    'attachment_id': att.get('attachment_id'),
                    'attachment_name': att_name,
                    'attachment_url': att.get('url'),
                    'local_filename': att.get('local_filename'),
                    'source_link_index': att.get('link_index'),
                    'fetch_status': att.get('fetch_status'),
                    'content_hash': content_hash,
                    'attachment_chunk_index': att_chunk_idx
                },
            })
            chunk_idx += 1

    return chunks_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_chunks:             list[dict] = []
    docs_processed:         int        = 0
    hard_cut_fixed:         int        = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc    = json.loads(line)
            chunks = build_chunks(doc)

            all_chunks.extend(chunks)
            docs_processed += 1

    # ── Count hard-cut fixes (post-merge applied inside build_chunks) ─────────
    for chunk in all_chunks:
        body_text = chunk['text'].split('\n', 2)[-1]   # skip prefix lines
        if body_text and body_text.lstrip() and body_text.lstrip()[0] in _VIET_MID_VOWELS:
            hard_cut_fixed += 1   # should be 0 after fix

    # ── Write output ─────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='\n') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # ── Statistics ────────────────────────────────────────────────────────────
    lengths     = [len(c['text']) for c in all_chunks]
    body_chunks = [c for c in all_chunks if not c['is_attachment']]
    att_chunks  = [c for c in all_chunks if c['is_attachment']]
    over_2000   = [c for c in all_chunks if len(c['text']) > 2000]
    tiny        = [c for c in all_chunks if len(c['text']) < 100]

    print(f"\n✅ Chunking hoàn tất! (LangChain RecursiveCharacterTextSplitter)")
    print(f"   Tổng docs đầu vào  : {docs_processed}")
    print(f"   Tổng chunks đầu ra : {len(all_chunks)}")
    print(f"     - Body chunks    : {len(body_chunks)}")
    print(f"     - Attach chunks  : {len(att_chunks)}")
    print(f"   Độ dài chunk:")
    print(f"     - Trung bình     : {sum(lengths)/len(lengths):.0f} chars")
    print(f"     - Lớn nhất       : {max(lengths)} chars")
    print(f"     - Nhỏ nhất       : {min(lengths)} chars")
    print(f"   Kiểm tra chất lượng:")
    print(f"     - Chunks vượt 2000 chars  : {len(over_2000)}")
    print(f"     - Chunks rất ngắn <100    : {len(tiny)}")
    print(f"     - Hard-cut còn sót (lý tưởng = 0): {hard_cut_fixed}")
    if over_2000:
        for c in over_2000[:5]:
            print(f"       ⚠  {c['id']} – {len(c['text'])} chars")
    if tiny:
        for c in tiny[:5]:
            print(f"       ⚠  {c['id']} – {len(c['text'])} chars | '{c['text'][:60]}'")
    print(f"\n   📄 Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
