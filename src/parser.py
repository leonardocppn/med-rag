"""
parser.py

Reads a PDF page by page, extracts every text element with its coordinates
(x0, y0, x1, y1) and classifies it into a logical region: header, footer,
title, table, or body.

pdfplumber provides for each word on the page:
  text, x0, y0, x1, y1, top, bottom, fontname, size

These are used to group words into lines, lines into paragraph blocks,
and each block is classified based on its vertical position and font size.

Layout parameters can be supplied externally via ParsingParams
(see profiler.py) to adapt to different document layouts.
"""

from __future__ import annotations

import pdfplumber
from dataclasses import dataclass


@dataclass
class Block:
    page: int
    region: str        # "title" | "header" | "footer" | "table" | "body"
    text: str
    top: float         # distance from the top edge of the page (points)
    font_size: float


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _classify_region(top: float, page_height: float,
                     font_size: float, median_size: float,
                     header_threshold: float = 0.08,
                     footer_threshold: float = 0.92,
                     title_font_ratio: float = 1.4) -> str:
    relative_pos = top / page_height

    if relative_pos < header_threshold:
        return "header"
    if relative_pos > footer_threshold:
        return "footer"
    if font_size >= median_size * title_font_ratio:
        return "title"
    return "body"


def _group_words_into_lines(words: list[dict],
                             line_gap: float = 3.0) -> list[list[dict]]:
    """Groups vertically adjacent words into the same line."""
    if not words:
        return []

    words = sorted(words, key=lambda w: (round(w["top"] / line_gap), w["x0"]))
    lines = []
    current_line = [words[0]]

    for word in words[1:]:
        last = current_line[-1]
        if abs(word["top"] - last["top"]) <= line_gap:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]

    lines.append(current_line)
    return lines


def _compute_adaptive_gap(lines: list[list[dict]],
                          multiplier: float = 2.0,
                          minimum: float = 10.0,
                          fallback: float = 12.0) -> float:
    """Computes the block separation gap based on the median spacing
    between consecutive lines on the page.
    Never goes below `minimum` to avoid excessive fragmentation."""
    if len(lines) < 2:
        return fallback

    gaps = []
    for i in range(len(lines) - 1):
        prev_bottom = max(w["bottom"] for w in lines[i])
        curr_top = min(w["top"] for w in lines[i + 1])
        gap = curr_top - prev_bottom
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return fallback

    median_gap = _median(gaps)
    return max(median_gap * multiplier, minimum)


def _lines_to_blocks(lines: list[list[dict]],
                     block_gap: float = 12.0) -> list[dict]:
    """Merges adjacent lines into paragraph blocks."""
    if not lines:
        return []

    blocks = []
    current_words = list(lines[0])

    for line in lines[1:]:
        prev_bottom = max(w["bottom"] for w in current_words)
        curr_top = min(w["top"] for w in line)
        if curr_top - prev_bottom <= block_gap:
            current_words.extend(line)
        else:
            blocks.append(current_words)
            current_words = list(line)

    blocks.append(current_words)
    return blocks


def _detect_columns(lines: list[list[dict]],
                    tolerance: float = 5.0,
                    min_ratio: float = 0.20) -> list[float]:
    """Finds recurring x0 positions that indicate columns.
    Returns x0 values that appear in at least min_ratio of the lines."""
    from collections import Counter

    if not lines:
        return []

    x0_values = [round(word["x0"] / tolerance) * tolerance
                 for line in lines
                 for word in line]

    counts = Counter(x0_values)
    threshold = len(lines) * min_ratio
    columns = sorted(x for x, count in counts.items() if count >= threshold)
    return columns


def _format_line_with_columns(line: list[dict],
                              columns: list[float]) -> str:
    """Formats a line by assigning words to the nearest column.
    Returns columns separated by ' | '."""
    if not columns:
        return " ".join(w["text"] for w in line)

    col_texts: dict[int, list[str]] = {}
    for word in line:
        col_idx = min(range(len(columns)),
                      key=lambda i: abs(word["x0"] - columns[i]))
        col_texts.setdefault(col_idx, []).append(word["text"])

    parts = []
    for idx in sorted(col_texts):
        parts.append(" ".join(col_texts[idx]))
    return " | ".join(parts)


def _format_block_text(block_words: list[dict],
                       lines: list[list[dict]],
                       columns: list[float],
                       min_columns_for_table: int = 3) -> str:
    """Formats a block's text. If the page has enough columns,
    the block's lines are formatted as a structured table.
    Otherwise, words are concatenated normally."""
    if len(columns) < min_columns_for_table:
        return " ".join(w["text"] for w in block_words).strip()

    # Reconstruct lines within this block
    block_lines = _group_words_into_lines(block_words)
    formatted = []
    for line in block_lines:
        formatted.append(_format_line_with_columns(line, columns))
    return "\n".join(formatted)


def _clean_text(text: str) -> str:
    """Removes layout artifacts from extracted text:
    lines of only dashes, isolated dots."""
    import re
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Remove lines of only dashes/dots/pipes
        if re.match(r'^[\s.\-|]+$', line):
            continue
        # Remove isolated dots in columns (". |" or "| .")
        line = re.sub(r'\.\s*\|', '|', line)
        line = re.sub(r'\|\s*\.(\s*\|)', r'|\1', line)
        line = re.sub(r'\|\s*\.\.?\s*$', '', line)
        line = re.sub(r'^\.\s*\|', '|', line)
        # Remove empty residual pipes and multiple spaces
        line = re.sub(r'\|\s*\|', '|', line)
        line = re.sub(r'^\s*\|\s*', '', line)
        line = re.sub(r'\s*\|\s*$', '', line)
        line = re.sub(r'\s{2,}', ' ', line).strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def extract_blocks(pdf_path: str,
                   header_threshold: float = 0.08,
                   footer_threshold: float = 0.92,
                   title_font_ratio: float = 1.4,
                   line_gap: float = 3.0,
                   block_gap_multiplier: float = 2.0,
                   block_gap_minimum: float = 10.0,
                   min_columns_for_table: int = 3) -> list[Block]:
    """
    Reads the PDF and returns a list of Blocks sorted by page
    and vertical position.

    Layout parameters can be customized to adapt to different
    document types (see profiler.py).
    """
    result = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(
                extra_attrs=["fontname", "size"],
                keep_blank_chars=False,
            )
            if not words:
                continue

            page_height = page.height

            # Median font size on the page (used to detect titles)
            sizes = [w.get("size", 10) or 10 for w in words]
            median_size = _median(sizes)

            lines = _group_words_into_lines(words, line_gap=line_gap)
            block_gap = _compute_adaptive_gap(
                lines, multiplier=block_gap_multiplier,
                minimum=block_gap_minimum,
            )
            columns = _detect_columns(lines)
            raw_blocks = _lines_to_blocks(lines, block_gap=block_gap)

            for block_words in raw_blocks:
                text = _clean_text(_format_block_text(
                    block_words, lines, columns,
                    min_columns_for_table=min_columns_for_table,
                ))
                if not text:
                    continue

                top = min(w["top"] for w in block_words)
                font_size = _median([w.get("size", 10) or 10
                                     for w in block_words])
                region = _classify_region(
                    top, page_height, font_size, median_size,
                    header_threshold=header_threshold,
                    footer_threshold=footer_threshold,
                    title_font_ratio=title_font_ratio,
                )

                result.append(Block(
                    page=page_num,
                    region=region,
                    text=text,
                    top=top,
                    font_size=font_size,
                ))

    return result



def extract_blocks_plain(pdf_path: str) -> list[Block]:
    """
    Plain-text extraction: reads the PDF page by page and returns one
    Block per page with region="body". No layout analysis, no column
    detection, no header/footer classification.

    Best for simple one-column documents or scanned/OCR PDFs where
    the layout structure is unreliable.
    """
    result = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            result.append(Block(
                page=page_num,
                region="body",
                text=text.strip(),
                top=0.0,
                font_size=0.0,
            ))

    return result


def blocks_to_chunks(blocks: list[Block],
                     skip_regions: set[str] | None = None,
                     overlap: int = 1) -> list[dict]:
    """
    Converts Blocks into chunks ready for indexing.
    Each chunk has: text, page, region.
    Header and footer blocks are discarded by default.

    overlap: number of preceding blocks to include as context,
    useful for capturing information at paragraph boundaries.
    """
    if skip_regions is None:
        skip_regions = {"header", "footer"}

    filtered = [
        b for b in blocks
        if b.region not in skip_regions and len(b.text.split()) >= 4
    ]

    chunks = []
    for i, b in enumerate(filtered):
        prefix_blocks = filtered[max(0, i - overlap):i]
        parts = [pb.text for pb in prefix_blocks] + [b.text]
        text = "\n\n".join(parts)

        chunks.append({
            "text": text,
            "page": b.page,
            "region": b.region,
        })

    return chunks
