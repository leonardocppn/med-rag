"""
parser.py

Reads a PDF page by page, extracts every text element with its coordinates
(x0, y0, x1, y1) and classifies it into a logical region: header, footer,
title, or body.

pdfplumber provides for each word on the page:
  text, x0, y0, x1, y1, top, bottom, fontname, size

These are used to group words into lines, lines into paragraph blocks,
and each block is classified based on its vertical position and font size.
"""

import pdfplumber
from dataclasses import dataclass


@dataclass
class Block:
    page: int
    region: str        # "title" | "header" | "footer" | "body"
    text: str
    top: float         # distance from the top edge of the page (points)
    font_size: float


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _classify_region(top: float, page_height: float,
                     font_size: float, median_size: float) -> str:
    relative_pos = top / page_height

    if relative_pos < 0.08:
        return "header"
    if relative_pos > 0.92:
        return "footer"
    if font_size >= median_size * 1.4:
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


def extract_blocks(pdf_path: str) -> list[Block]:
    """
    Reads the PDF and returns a list of Blocks sorted by page
    and vertical position.
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

            lines = _group_words_into_lines(words)
            raw_blocks = _lines_to_blocks(lines)

            for block_words in raw_blocks:
                text = " ".join(w["text"] for w in block_words).strip()
                if not text:
                    continue

                top = min(w["top"] for w in block_words)
                font_size = _median([w.get("size", 10) or 10
                                     for w in block_words])
                region = _classify_region(
                    top, page_height, font_size, median_size
                )

                result.append(Block(
                    page=page_num,
                    region=region,
                    text=text,
                    top=top,
                    font_size=font_size,
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
        chunks.append({
            "text": "\n\n".join(parts),
            "page": b.page,
            "region": b.region,
        })

    return chunks
