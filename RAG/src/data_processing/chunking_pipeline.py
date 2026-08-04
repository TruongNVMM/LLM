#!/usr/bin/env python3
"""
chunking_pipeline.py – Smart chunking for HUST Sổ tay RAG documents.

Strategy:
1. ATTACHMENT EXTRACTION: Each [Nội dung tài liệu đính kèm: X] block becomes
   an independent chunk (with its own context prefix), then recursive-splits if
   the block is still longer than CHUNK_SIZE.
2. BODY CHUNKING: After stripping attachments, the remaining article body is
   split recursively using separator priority:
     \n\n  >  \n  >  .  >  ,  >  space
   This preserves Markdown table rows (| col | col |) intact.
3. CONTEXT PRESERVATION: Every chunk carries:
   - A 1-2 line prefix: "Bài: <title> | Chuyên mục: <category>"
   - Full metadata inherited from parent doc
   - chunk_index, chunk_type (body | attachment), is_attachment flag

Output:
  data/processed/rag_chunks.jsonl
"""

import json
import re
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1500       # target chars per chunk (≈ 300-400 tokens)
CHUNK_OVERLAP = 200     # overlap chars between consecutive body chunks
MAX_ATTACHMENT_CHARS = 10_000  # cap each attachment chunk before recursive split
INPUT_FILE = Path('data/processed/rag_documents.jsonl')
OUTPUT_FILE = Path('data/processed/rag_chunks.jsonl')

# Separators in priority order – we try each level before breaking mid-word
SEPARATORS = ['\n\n', '\n', '。', '. ', ', ', ' ', '']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recursive_split(text: str, size: int, overlap: int,
                    separators: list[str] | None = None) -> list[str]:
    """Split text into chunks of at most `size` chars, with `overlap`."""
    if separators is None:
        separators = SEPARATORS

    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    # Try each separator level
    for sep in separators:
        if sep == '':
            # Last resort: hard cut
            chunks, start = [], 0
            while start < len(text):
                end = min(start + size, len(text))
                chunks.append(text[start:end])
                start += size - overlap
            return chunks

        if sep not in text:
            continue

        parts = text.split(sep)
        chunks, current = [], ''
        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) > size and current:
                # Flush current chunk, then recurse on 'part' if it's too big
                chunks.append(current.strip())
                tail = (text[text.rfind(current) + len(current):]).lstrip(sep)
                # Overlap: carry last `overlap` chars into next chunk
                carry = current[-overlap:] if len(current) > overlap else current
                current = (carry + sep + part).lstrip(sep)
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())

        # Recursively split any chunk that's still too large
        final = []
        for c in chunks:
            if len(c) > size:
                final.extend(recursive_split(c, size, overlap, separators[separators.index(sep)+1:]))
            else:
                final.append(c)
        return final

    return [text]


ATTACH_PATTERN = re.compile(
    r'\[Nội dung tài liệu đính kèm: (?P<name>[^\]]+)\]'
    r'(?P<content>.*?)'
    r'\[/Nội dung\]',
    re.DOTALL
)

def _clean_attachment_name(name: str) -> str:
    """Return a human-readable attachment label."""
    name = name.strip()
    # If it looks like a URL, shorten it
    if name.startswith('http'):
        return 'Tài liệu đính kèm (Google Drive/Doc)'
    # Remove trailing punctuation artefacts  (e.g. "Xem tại đây.")
    if re.fullmatch(r'[Xx]em tại đây[.,;]?', name):
        return 'Tài liệu đính kèm'
    return name


def extract_attachments(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Strip attachment blocks from text.
    Returns (clean_body, [(attachment_name, attachment_content), ...])
    """
    attachments = []
    def _replace(m: re.Match) -> str:
        attachments.append((_clean_attachment_name(m.group('name')),
                            m.group('content').strip()))
        return ''
    body = ATTACH_PATTERN.sub(_replace, text)
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    return body, attachments


def build_chunks(doc: dict) -> list[dict]:
    base_meta = dict(doc['metadata'])
    title = doc['title']
    cat = doc['metadata'].get('category_name', '')
    prefix = f"Bài viết: {title} | Chuyên mục: {cat}\n"

    text = doc['text']
    body, attachments = extract_attachments(text)
    chunks_out = []
    chunk_idx = 0

    # ── Body chunks ─────────────────────────────────────────────────────────
    body_parts = recursive_split(body, CHUNK_SIZE, CHUNK_OVERLAP)
    for part in body_parts:
        if not part.strip():
            continue
        chunks_out.append({
            'id': f"{doc['id']}_chunk{chunk_idx:03d}",
            'parent_id': doc['id'],
            'chunk_index': chunk_idx,
            'chunk_type': 'body',
            'is_attachment': False,
            'attachment_name': None,
            'text': prefix + part,
            'metadata': {**base_meta,
                         'chunk_index': chunk_idx,
                         'chunk_type': 'body',
                         'is_attachment': False}
        })
        chunk_idx += 1

    # ── Attachment chunks ────────────────────────────────────────────────────
    for att_name, att_content in attachments:
        att_content = att_content[:MAX_ATTACHMENT_CHARS]  # hard cap
        att_parts = recursive_split(att_content, CHUNK_SIZE, CHUNK_OVERLAP)
        att_prefix = f"Bài viết: {title} | Chuyên mục: {cat}\nBiểu mẫu/Tài liệu đính kèm: {att_name}\n"
        for part in att_parts:
            if not part.strip():
                continue
            chunks_out.append({
                'id': f"{doc['id']}_chunk{chunk_idx:03d}",
                'parent_id': doc['id'],
                'chunk_index': chunk_idx,
                'chunk_type': 'attachment',
                'is_attachment': True,
                'attachment_name': att_name,
                'text': att_prefix + part,
                'metadata': {**base_meta,
                             'chunk_index': chunk_idx,
                             'chunk_type': 'attachment',
                             'is_attachment': True,
                             'attachment_name': att_name}
            })
            chunk_idx += 1

    return chunks_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_chunks = []
    docs_processed = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            chunks = build_chunks(doc)
            all_chunks.extend(chunks)
            docs_processed += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='\n') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # ── Stats ────────────────────────────────────────────────────────────────
    lengths = [len(c['text']) for c in all_chunks]
    body_chunks = [c for c in all_chunks if not c['is_attachment']]
    att_chunks  = [c for c in all_chunks if c['is_attachment']]

    print(f"✅ Chunking hoàn tất!")
    print(f"   Tổng docs đầu vào : {docs_processed}")
    print(f"   Tổng chunks đầu ra: {len(all_chunks)}")
    print(f"     - Body chunks   : {len(body_chunks)}")
    print(f"     - Attach chunks : {len(att_chunks)}")
    print(f"   Độ dài chunk:")
    print(f"     - Trung bình    : {sum(lengths)/len(lengths):.0f} chars")
    print(f"     - Lớn nhất      : {max(lengths)} chars")
    print(f"     - Nhỏ nhất      : {min(lengths)} chars")
    over = [c for c in all_chunks if len(c['text']) > 2000]
    print(f"   Chunks vượt 2000 chars: {len(over)}")
    if over:
        for c in over[:5]:
            print(f"     ⚠  {c['id']} – {len(c['text'])} chars")
    print(f"\n   📄 Output: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
