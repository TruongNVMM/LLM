#!/usr/bin/env python3
"""Crawl HUST student handbook data for RAG preprocessing.

The sv-ctt.hust.edu.vn page is a Vue SPA. Its handbook route fetches data from
ctsv.hust.edu.vn/api-t, so this crawler uses the same public JSON endpoints and
then converts each HTML description into text plus structured link metadata.

Optionally (--fetch-attachments), the script also downloads and extracts text
from linked documents:
  - Google Doc  → export?format=txt (requires public access)
  - Google Drive / PDF  → download then pypdf (requires pypdf)
  - SharePoint Word Online → requires authentication, skipped automatically
"""

from __future__ import annotations

import logging
import warnings

import argparse
import io
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass

try:
    from markdownify import markdownify as md
except ImportError:
    md = None  # type: ignore
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

try:
    from pypdf import PdfReader as _PdfReader  # pypdf >= 3.x
    # Silence pypdf's verbose warnings about malformed PDFs
    logging.getLogger("pypdf").setLevel(logging.ERROR)
except ImportError:
    try:
        from PyPDF2 import PdfReader as _PdfReader  # type: ignore[no-redef]
        logging.getLogger("PyPDF2").setLevel(logging.ERROR)
    except ImportError:
        _PdfReader = None  # type: ignore[assignment,misc]

# Magic-byte signatures for non-PDF binary formats we might receive
_NON_PDF_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG",       # PNG image
    b"\xff\xd8\xff",  # JPEG image
    b"GIF8",          # GIF image
    b"PK\x03\x04",   # ZIP / DOCX / XLSX
    b"\xd0\xcf\x11\xe0",  # OLE2 / old .doc / .xls
)


API_BASE = "https://ctsv.hust.edu.vn/api-t"
PAGE_BASE = "https://sv-ctt.hust.edu.vn/#/so-tay-sv"
DEFAULT_OUTPUT_DIR = Path("data") / "hust_sotay"

# Attachment fetching
ATTACHMENT_TIMEOUT = 25          # seconds per request
ATTACHMENT_MAX_CHARS = 10_000    # truncate very long attachments
ATTACHMENT_FETCH_DELAY = 0.4     # polite delay between requests
# Types we can fetch automatically (no auth required)
FETCHABLE_TYPES = {"google_doc", "google_drive", "pdf"}

CATEGORIES = {
    "HT": {
        "name": "Học tập",
        "desc": "Đăng ký học, kết quả, học lại, xét tốt nghiệp và các vấn đề học vụ.",
        "order": 1,
    },
    "HC": {
        "name": "Thủ tục hành chính",
        "desc": "Giấy tờ, xác nhận, đơn từ và các thủ tục hành chính cho sinh viên.",
        "order": 2,
    },
    "HB": {
        "name": "Học bổng & Hỗ trợ tài chính",
        "desc": "Học bổng khuyến khích, hỗ trợ tài chính và miễn giảm học phí.",
        "order": 3,
    },
    "DS": {
        "name": "Đời sống & Hỗ trợ sinh viên",
        "desc": "BHYT, ký túc xá, vé xe buýt và các hỗ trợ đời sống sinh viên.",
        "order": 4,
    },
    "KN": {
        "name": "Kỹ năng và hoạt động sinh viên",
        "desc": "Tài khoản, ứng dụng số của trường và kỹ năng cần thiết.",
        "order": 5,
    },
    "HD": {
        "name": "Tốt nghiệp và việc làm",
        "desc": "Hoạt động ngoại khóa, rèn luyện và thủ tục tốt nghiệp.",
        "order": 6,
    },
}

TYPE_DOC_TO_CATEGORY = {
    "Sổ tay SV@HT": "HT",
    "Sổ tay SV@HC": "HC",
    "Sổ tay SV@HB": "HB",
    "Sổ tay SV@ĐS": "DS",
    "Sổ tay SV@KN": "KN",
    "Sổ tay SV@HD": "HD",
}

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

FILE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".pdf",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
}


@dataclass
class LinkSpan:
    href: str
    text: str
    start: int
    end: int


""" Class HtmlTextAndLinkParser extends HTMLParser to extract text and links from HTML content """
# VD: <p>Hello <a href="/abc">world</a></p>
# Code sẽ gọi theo thứ tự: 
    # handle_starttag("p")
    # handle_data("Hello ")

    # handle_starttag("a")
    # handle_data("world")
    # handle_endtag("a")

    # handle_endtag("p")

class HtmlTextAndLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []                          # Lưu các đoạn text: parts = ["Hello ", "world"]
        self.links: list[LinkSpan] = []                     # Lưu danh sách các liên kết links = [LinkSpan(href="/abc", text="world", start=6, end=11)]
        self._active_link: dict[str, Any] | None = None     # Lưu thông tin liên kết đang xử lý, ví dụ: {"href": "/abc", "start": 6, "parts": ["world"]}

    # @property để dùng như một thuộc tính chứ không phải là một phương thức gọi hàm: HtmlTextAndLinkParser.text_so_far 
    # Trả về chuỗi text đã parse đến hiện tại
    @property
    def text_so_far(self) -> str:
        return "".join(self.parts)

    # Hàm handle_starttag được gọi khi parser gặp một thẻ mở (ví dụ: <a href="/abc">)
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in BLOCK_TAGS:
            self._append_newline()
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self._active_link = {"href": href, "start": len(self.text_so_far), "parts": []}

    # Hàm handle_endtag được gọi khi parser gặp một thẻ đóng (ví dụ: </a>)
    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._active_link is not None:
            link_text = clean_text("".join(self._active_link["parts"]))
            self.links.append(
                LinkSpan(
                    href=self._active_link["href"],
                    text=link_text,
                    start=self._active_link["start"],
                    end=len(self.text_so_far),
                )
            )
            self._active_link = None
        if tag in BLOCK_TAGS:
            self._append_newline()

    # Hàm handle_data được gọi khi parser gặp một đoạn text (ví dụ: "Hello ")
    def handle_data(self, data: str) -> None:
        if not data:
            return
        self.parts.append(data)
        if self._active_link is not None:
            self._active_link["parts"].append(data)

    # Hàm _append_newline được gọi để thêm một dòng mới vào parts nếu cần thiết
    def _append_newline(self) -> None:
        if self.parts and not self.text_so_far.endswith("\n"):
            self.parts.append("\n")

""" Function to send a POST request with JSON data """
# endpoint: đường dẫn API, ví dụ "login" hoặc "users/create"
# payload: dữ liệu sẽ gửi lên server dưới dạng dictionary
# timeout: thời gian chờ tối đa là 30 giây
def post_json(endpoint: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")          # Chuyển payload từ dict sang json
    request = Request(
        f"{API_BASE}/{endpoint.lstrip('/')}",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "hust-sotay-rag-crawler/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Request failed for {endpoint}: {exc}") from exc


""" Attachment fetching """

def _http_get_bytes(url: str) -> bytes | None:
    """Download raw bytes from a URL; return None on any error."""
    try:
        req = Request(
            url,
            headers={"User-Agent": "HUST-RAG-crawler/1.1"},
        )
        with urlopen(req, timeout=ATTACHMENT_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return None


def _is_pdf(data: bytes) -> bool:
    """Return True if data looks like a PDF (starts with %PDF magic bytes)."""
    return data[:4] == b"%PDF"


def _pdf_bytes_to_text(data: bytes) -> str:
    """
    Extract plain text from PDF bytes using pypdf.
    Returns '' if unavailable, not a PDF, or no text could be extracted.
    """
    if _PdfReader is None or not _is_pdf(data):
        return ""
    # Guard against non-PDF magic bytes that would cause noisy warnings
    if any(data[:4] == sig[:4] for sig in _NON_PDF_MAGIC):
        return ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = _PdfReader(io.BytesIO(data))
            parts = [page.extract_text() or "" for page in reader.pages]
        return clean_text("\n".join(parts))
    except Exception:
        return ""


def _extract_gdoc_id(url: str) -> str | None:
    m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def _extract_gdrive_id(url: str) -> str | None:
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def fetch_attachment_text(link: dict[str, Any]) -> tuple[str | None, str]:
    """
    Attempt to download and extract text from a linked document.

    Returns:
        (text, status) where status is one of:
          'ok'            – text extracted successfully
          'empty'         – downloaded but no text could be extracted
          'fetch_error'   – network / HTTP error
          'no_pypdf'      – PDF but pypdf not installed
          'auth_required' – SharePoint / Word Online (needs login)
          'skipped'       – link type not fetchable
    """
    link_type = link.get("type", "")
    url = link.get("url", "")

    # SharePoint / Word Online requires Microsoft 365 SSO – skip gracefully
    if link_type in ("word_online", "excel_online", "powerpoint_online"):
        return None, "auth_required"

    if link_type not in FETCHABLE_TYPES:
        return None, "skipped"

    # Google Doc 
    if link_type == "google_doc":
        doc_id = _extract_gdoc_id(url)
        if not doc_id:
            return None, "fetch_error"
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        data = _http_get_bytes(export_url)
        if data is None:
            return None, "fetch_error"
        text = clean_text(data.decode("utf-8", errors="replace"))
        if not text:
            return None, "empty"
        return text[:ATTACHMENT_MAX_CHARS], "ok"

    # Google Drive (usually PDF) 
    if link_type == "google_drive":
        file_id = _extract_gdrive_id(url)
        if not file_id:
            return None, "fetch_error"
        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        data = _http_get_bytes(dl_url)
        if data is None:
            return None, "fetch_error"
        # Reject non-PDF files (images, HTML virus-warning pages, etc.)
        if not _is_pdf(data):
            if any(data[:4] == sig[:4] for sig in _NON_PDF_MAGIC):
                return None, "empty"   # image/binary file, not extractable
            if data[:100].lower().lstrip().startswith(b"<!doc") or b"<html" in data[:200].lower():
                return None, "fetch_error"  # Google large-file warning page
            return None, "empty"
        text = _pdf_bytes_to_text(data)
        if text:
            return text[:ATTACHMENT_MAX_CHARS], "ok"
        return None, "empty"

    # Direct PDF link
    if link_type == "pdf":
        if _PdfReader is None:
            return None, "no_pypdf"
        data = _http_get_bytes(url)
        if data is None:
            return None, "fetch_error"
        text = _pdf_bytes_to_text(data)
        if text:
            return text[:ATTACHMENT_MAX_CHARS], "ok"
        return None, "empty"

    return None, "skipped"


def enrich_links(
    links: list[dict[str, Any]],
    *,
    fetch: bool,
    delay: float = ATTACHMENT_FETCH_DELAY,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    For each link, optionally fetch attachment content and annotate with
    'content' (str | None) and 'fetch_status' ('ok' | 'auth_required' | ...).
    """
    enriched: list[dict[str, Any]] = []
    fetched_count = 0
    for link in links:
        entry = dict(link)
        if fetch and link.get("type") in FETCHABLE_TYPES:
            if fetched_count > 0 and delay > 0:
                time.sleep(delay)
            text, status = fetch_attachment_text(link)
            entry["content"] = text
            entry["fetch_status"] = status
            fetched_count += 1
            if verbose:
                mark = "✓" if status == "ok" else "✗"
                print(f"    [{mark}] {link['type']} {status}: {link['url'][:60]}", flush=True)
        else:
            entry["content"] = None
            entry["fetch_status"] = "auth_required" if link.get("type") in ("word_online", "excel_online", "powerpoint_online") else "skipped"
        enriched.append(entry)
    return enriched


def get_meaningful_attachment_name(link: dict[str, Any]) -> str:
    local_name = str(link.get("local_filename") or "").strip()
    if local_name and not local_name.startswith("http") and not is_generic_anchor(local_name):
        return local_name

    anchor = str(link.get("anchor_text") or "").strip()
    if anchor and not is_generic_anchor(anchor) and len(anchor) < 80:
        return anchor
        
    context = str(link.get("context") or "").strip()
    if context:
        lines = [line.strip() for line in context.split("\n") if line.strip()]
        for line in lines:
            clean_line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line)
            clean_line = re.sub(r"[#\*\|_]+", "", clean_line).strip()
            # Bỏ từ bị cắt dở ở đầu dòng (ví dụ: 'ghiệp; lịch...' -> 'lịch...')
            clean_line = re.sub(r"^[^\s\w]*[a-zA-Z0-9àáảãạăắằẳẵặâấầnẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+[^\s\w]*\s+", "", clean_line).strip()
            if clean_line and not is_generic_anchor(clean_line) and len(clean_line) > 3:
                if "@" not in clean_line and not clean_line.isdigit():
                    return clean_line[:80]
                    
    return "Tài liệu đính kèm"


def build_attachment_section(links: list[dict[str, Any]]) -> str:
    """Build a readable section of fetched attachment content for rag_text."""
    sections: list[str] = []
    for link in links:
        content = link.get("content")
        if not content:
            continue
        label = get_meaningful_attachment_name(link)
        sections.append(
            f"[Nội dung tài liệu đính kèm: {label}]\n{content}\n[/Nội dung]"
        )
    return "\n\n".join(sections)


def clean_emojis_and_symbols(value: str) -> str:
    if not value:
        return ""
    cleaned = []
    for ch in value:
        cat = unicodedata.category(ch)
        # Strip control characters (Cc) except newline, carriage return, and tab
        if cat == "Cc" and ch not in ("\n", "\r", "\t"):
            continue
        if cat == "So" or ch in ("□", "☐", "■", "▪", "▫", "♦", "●", "○", "★", "☆", "▶", "►", "◄", "▼", "▲"):
            continue
        cleaned.append(ch)
    res = "".join(cleaned)
    res = re.sub(r" +", " ", res)
    return res.strip()


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = clean_emojis_and_symbols(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "khong-co-tieu-de"

def extract_keywords(text: str) -> dict[str, list[str]]:
    # 1. Extract emails
    emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)))
    
    # 2. Extract tags in brackets [Tag]
    # Ignore markdown links [text](url) and our own system tags [Nội dung...]/[Biểu mẫu...]
    raw_tags = re.findall(r'\[([^\]]+)\](?!\()', text)
    tags = []
    for tag in raw_tags:
        clean_tag = tag.strip()
        if not clean_tag.startswith("Nội dung tài liệu") and not clean_tag.startswith("Biểu mẫu đính kèm"):
            # Avoid single character brackets or too long sentences
            if 1 < len(clean_tag) < 50:
                tags.append(clean_tag)
            
    return {
        "emails": emails,
        "tags": list(set(tags))
    }

def detect_link_type(url: str) -> str:
    lower = url.lower().split("?", 1)[0].split("#", 1)[0]
    for ext in FILE_EXTENSIONS:
        if lower.endswith(ext):
            return ext.lstrip(".")
    if url.startswith("mailto:"):
        return "email"
    if "forms.office.com" in lower or "docs.google.com/forms" in lower:
        return "form"
    if "sharepoint.com/:w:/" in lower or "sharepoint.com/:w:/" in url.lower():
        return "word_online"
    if "sharepoint.com/:x:/" in lower or "sharepoint.com/:x:/" in url.lower():
        return "excel_online"
    if "sharepoint.com/:p:/" in lower or "sharepoint.com/:p:/" in url.lower():
        return "powerpoint_online"
    if "docs.google.com/document" in lower:
        return "google_doc"
    if "docs.google.com/spreadsheets" in lower:
        return "google_sheet"
    if "drive.google.com" in lower:
        return "google_drive"
    if lower.startswith("http://") or lower.startswith("https://"):
        return "html_or_external"
    return "relative_or_unknown"


def is_generic_anchor(anchor: str) -> bool:
    value = unicodedata.normalize("NFD", anchor or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D").lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value in {"tai day", "link", "xem them", "bam vao day", "click here"}


def extract_text_and_links(html: str, base_url: str, title: str) -> tuple[str, list[dict[str, Any]]]:
    parser = HtmlTextAndLinkParser()
    parser.feed(html or "")
    raw_text = parser.text_so_far
    text = clean_text(raw_text)
    links = []
    for idx, span in enumerate(parser.links, start=1):
        absolute_url = urljoin(base_url, span.href)
        start_pos = max(0, span.start - 180)
        # Nếu start_pos rơi vào giữa từ, lùi lại/tiến lên khoảng trắng gần nhất
        if start_pos > 0 and not raw_text[start_pos - 1].isspace():
            space_idx = raw_text.find(" ", start_pos)
            if space_idx != -1 and space_idx < span.start:
                start_pos = space_idx + 1

        end_pos = min(len(raw_text), span.end + 180)
        context = clean_text(raw_text[start_pos:end_pos])
        anchor = span.text or absolute_url
        if is_generic_anchor(anchor):
            meaning = f"Liên kết trong bài '{title}': {context}"
        else:
            meaning = anchor
        links.append(
            {
                "index": idx,
                "anchor_text": anchor,
                "url": absolute_url,
                "raw_href": span.href,
                "type": detect_link_type(absolute_url),
                "context": context,
                "meaning": meaning,
            }
        )
    
    if md is not None:
        text = md(html or "", heading_style="ATX").strip()
        
    return text, links


def build_link_text(links: list[dict[str, Any]]) -> str:
    if not links:
        return ""
    lines = ["Liên kết liên quan:"]
    for link in links:
        lines.append(
            f"- {link['meaning']} | anchor='{link['anchor_text']}' | type={link['type']} | url={link['url']}"
        )
    return "\n".join(lines)


def normalize_article(
    raw: dict[str, Any],
    detail: dict[str, Any] | None,
    *,
    fetch_attachments: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    item = detail or raw
    doc_id = item.get("DocumentID") or raw.get("DocumentID")
    title = clean_text(str(item.get("Title") or raw.get("Title") or ""))
    type_doc = item.get("TypeDoc") or raw.get("TypeDoc")
    category_code = TYPE_DOC_TO_CATEGORY[type_doc]
    category = CATEGORIES[category_code]
    source_url = f"{PAGE_BASE}/{doc_id}/{slugify(title)}"
    html = item.get("Description") or ""
    text, raw_links = extract_text_and_links(html, source_url, title)

    # Optionally fetch attachment content for supported link types
    links = enrich_links(
        raw_links,
        fetch=fetch_attachments,
        verbose=verbose,
    )

    # Rename generic anchor texts in the markdown text for fetched attachments
    # e.g., [Tại đây](url) -> [Tài liệu đính kèm: URL](url)
    if fetch_attachments:
        for link in links:
            if link.get("content"):
                url = link.get("url", "")
                anchor = link.get("anchor_text", "")
                if is_generic_anchor(anchor):
                    # Guess a name from url or just say "Tài liệu đính kèm"
                    display_name = "Tài liệu đính kèm"
                    if "drive.google.com" in url or "docs.google.com" in url or url.endswith(".pdf"):
                        display_name = "Biểu mẫu/Tài liệu đính kèm"
                    
                    old_link = f"[{anchor}]({url})"
                    new_link = f"[{display_name}]({url})"
                    if old_link in text:
                        text = text.replace(old_link, new_link)
                    else:
                        # Fallback regex
                        escaped_url = re.escape(url)
                        text = re.sub(rf'\[([^\]]+)\]\({escaped_url}\)', rf'[{display_name}]({url})', text)

    link_text = build_link_text(links)
    attachment_section = build_attachment_section(links) if fetch_attachments else ""

    rag_text = clean_text(
        "\n".join(
            part
            for part in [
                f"Tiêu đề: {title}",
                f"Nhóm: {category['name']}",
                f"Loại tài liệu: {type_doc}",
                f"Cập nhật: {item.get('TimeCreate') or raw.get('TimeCreate') or ''}",
                text,
                link_text,
                attachment_section,
            ]
            if part
        )
    )
    return {
        "doc_id": doc_id,
        "title": title,
        "type_doc": type_doc,
        "category_code": category_code,
        "category_name": category["name"],
        "category_desc": category["desc"],
        "category_order": category["order"],
        "source_url": source_url,
        "api_detail_endpoint": f"{API_BASE}/HWAdmin/GetWebTitleInfo",
        "time_create": item.get("TimeCreate") or raw.get("TimeCreate"),
        "status": item.get("Status") if "Status" in item else raw.get("Status"),
        "creator_id": item.get("CreaterID") or raw.get("CreaterID"),
        "html": html,
        "text": text,
        "keywords": extract_keywords(text),
        "links": links,
        "rag_text": rag_text,
        "raw": item,
    }


def to_rag_document(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"hust_sotay_{article['doc_id']}",
        "doc_type": "hust_student_handbook_article",
        "title": article["title"],
        "text": article["rag_text"],
        "metadata": {
            "source": "HUST Sổ tay sinh viên",
            "source_url": article["source_url"],
            "doc_id": article["doc_id"],
            "type_doc": article["type_doc"],
            "category_code": article["category_code"],
            "category_name": article["category_name"],
            "time_create": article["time_create"],
            "emails": article.get("keywords", {}).get("emails", []),
            "tags": article.get("keywords", {}).get("tags", []),
            "links": article["links"],
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def crawl(
    output_dir: Path,
    use_detail_endpoint: bool,
    delay_seconds: float,
    fetch_attachments: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()

    if fetch_attachments and _PdfReader is None:
        print(
            "WARNING: pypdf not installed – PDF/Google Drive attachments will be skipped.\n"
            "         Run: pip install pypdf",
            file=sys.stderr,
        )

    listing = post_json("HWAdmin/GetWebTitleLst", {})
    if listing.get("RespCode") != 0:
        raise RuntimeError(f"API returned RespCode={listing.get('RespCode')}: {listing.get('RespText')}")

    raw_items = listing.get("WebTitleLst") or []
    handbook_items = [
        item for item in raw_items if item.get("TypeDoc") in TYPE_DOC_TO_CATEGORY and item.get("Status") == 1
    ]
    handbook_items.sort(key=lambda item: str(item.get("TimeCreate") or ""), reverse=True)

    articles: list[dict[str, Any]] = []
    for index, raw in enumerate(handbook_items, start=1):
        detail_item = None
        if use_detail_endpoint:
            detail = post_json("HWAdmin/GetWebTitleInfo", {"DocumentID": raw.get("DocumentID")})
            if detail.get("RespCode") != 0:
                raise RuntimeError(
                    f"Detail API returned RespCode={detail.get('RespCode')} for DocumentID={raw.get('DocumentID')}"
                )
            detail_item = detail.get("WebTitleInfo")
            if delay_seconds > 0 and index < len(handbook_items):
                time.sleep(delay_seconds)
        if verbose and fetch_attachments:
            print(f"  [{index}/{len(handbook_items)}] {raw.get('DocumentID')} – fetching attachments...", flush=True)
        articles.append(
            normalize_article(
                raw,
                detail_item,
                fetch_attachments=fetch_attachments,
                verbose=verbose,
            )
        )

    # Attachment fetch stats
    attachment_stats: dict[str, int] = {}
    for article in articles:
        for link in article.get("links", []):
            s = link.get("fetch_status", "skipped")
            attachment_stats[s] = attachment_stats.get(s, 0) + 1

    rag_documents = [to_rag_document(article) for article in articles]
    summary = {
        "source_page": PAGE_BASE,
        "api_base": API_BASE,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_api_items": len(raw_items),
        "total_handbook_articles": len(articles),
        "fetch_attachments": fetch_attachments,
        "attachment_stats": attachment_stats,
        "categories": CATEGORIES,
        "type_doc_to_category": TYPE_DOC_TO_CATEGORY,
        "outputs": {
            "raw_listing": str(output_dir / "raw" / "web_title_list.json"),
            "articles_json": str(output_dir / "processed" / "sotay_articles.json"),
            "articles_jsonl": str(output_dir / "processed" / "sotay_articles.jsonl"),
            "rag_jsonl": str(output_dir / "processed" / "rag_documents.jsonl"),
        },
    }

    write_json(output_dir / "raw" / "web_title_list.json", listing)
    write_json(output_dir / "processed" / "sotay_articles.json", articles)
    write_jsonl(output_dir / "processed" / "sotay_articles.jsonl", articles)
    write_jsonl(output_dir / "processed" / "rag_documents.jsonl", rag_documents)
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl HUST Sổ tay sinh viên data for RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crawl_hust_sotay.py
      Crawl articles only (no attachment fetching)

  python crawl_hust_sotay.py --fetch-attachments
      Crawl + download Google Doc / Google Drive / PDF attachments

  python crawl_hust_sotay.py --fetch-attachments --verbose
      Same, with per-link progress output

Note: SharePoint/Word Online links require Microsoft 365 login and are
always skipped. Install pypdf for PDF support: pip install pypdf
Install markdownify for better table preservation: pip install markdownify
    """,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Use descriptions from GetWebTitleLst only (skip GetWebTitleInfo per article).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay in seconds between detail API calls (default: 0.15).",
    )
    parser.add_argument(
        "--fetch-attachments",
        action="store_true",
        help=(
            "Download and extract text from linked Google Doc, Google Drive, "
            "and direct PDF attachments. Requires pypdf for PDFs. "
            "SharePoint links are al ways skipped (require auth)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-link fetch results when --fetch-attachments is active.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    try:
        summary = crawl(
            output_dir=args.output_dir,
            use_detail_endpoint=not args.no_detail,
            delay_seconds=max(args.delay, 0),
            fetch_attachments=args.fetch_attachments,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
