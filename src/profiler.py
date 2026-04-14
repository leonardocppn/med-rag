"""
profiler.py

Analyzes a PDF and extracts purely numerical layout metrics,
from which adaptive parsing parameters are derived. No hardcoded
textual signals: works with any layout, even unseen ones.

Three responsibilities:
  1. Metric extraction   — profile_pdf()
  2. Parameter derivation — derive_params() (adaptive per PDF)
  3. Persistence & clustering — ProfileStore (cross-document type discovery)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pdfplumber


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class LayoutProfile:
    """Purely numerical layout metrics extracted from a PDF."""
    pdf_path: str = ""
    num_pages: int = 0
    words_per_page: float = 0.0

    # Font
    font_size_median: float = 0.0
    font_size_std: float = 0.0
    font_size_levels: int = 0       # number of distinct size levels

    # Spacing
    line_gap_median: float = 0.0
    line_gap_std: float = 0.0
    block_gap_median: float = 0.0   # gap between blocks (> adaptive threshold)

    # Page structure
    column_density: float = 0.0     # ratio of multi-column lines / total
    header_zone: float = 0.0        # relative position of header end (0-1)
    footer_zone: float = 0.0        # relative position of footer start (0-1)
    content_density: float = 0.0    # words in body zone / total words

    # Titles
    title_font_gap: float = 0.0     # natural gap between body and title fonts

    def feature_vector(self) -> np.ndarray:
        """Returns the feature vector for clustering."""
        return np.array([
            self.words_per_page,
            self.font_size_median,
            self.font_size_std,
            self.font_size_levels,
            self.line_gap_median,
            self.line_gap_std,
            self.block_gap_median,
            self.column_density,
            self.header_zone,
            self.footer_zone,
            self.content_density,
            self.title_font_gap,
        ], dtype=np.float64)

    FEATURE_NAMES = [
        "words_per_page", "font_size_median", "font_size_std",
        "font_size_levels", "line_gap_median", "line_gap_std",
        "block_gap_median", "column_density", "header_zone",
        "footer_zone", "content_density", "title_font_gap",
    ]


@dataclass
class ParsingParams:
    """Parsing parameters derived from layout metrics."""
    header_threshold: float = 0.08
    footer_threshold: float = 0.92
    title_font_ratio: float = 1.4
    line_gap: float = 3.0
    block_gap_multiplier: float = 2.0
    block_gap_minimum: float = 10.0
    chunk_overlap: int = 1
    min_columns_for_table: int = 3
    skip_regions: set[str] = field(default_factory=lambda: {"header", "footer"})


# ── Utilities ───────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


# ── Metric extraction ───────────────────────────────────────────────────────

def _find_content_boundaries(words_by_page: list[list[dict]],
                             page_heights: list[float],
                             scan_zone: float = 0.25,
                             min_gap_ratio: float = 0.02
                             ) -> tuple[float, float]:
    """Finds header/footer boundaries by analyzing content gaps.

    Scans the top and bottom `scan_zone` of each page looking for the
    largest vertical gap: that marks the boundary between header/footer
    and body. Works on relative positions (0-1) aggregated across all pages.
    """
    if not words_by_page:
        return 0.08, 0.92

    rel_positions: list[float] = []
    for words, height in zip(words_by_page, page_heights):
        if height == 0:
            continue
        for w in words:
            rel_positions.append(w["top"] / height)

    if len(rel_positions) < 10:
        return 0.08, 0.92

    rel_positions.sort()

    # --- Header boundary: largest gap in top scan_zone ---
    header_end = 0.08
    top_pos = [p for p in rel_positions if p < scan_zone]
    if len(top_pos) >= 2:
        best_gap, best_boundary = 0.0, header_end
        for i in range(len(top_pos) - 1):
            gap = top_pos[i + 1] - top_pos[i]
            if gap > best_gap:
                best_gap = gap
                best_boundary = top_pos[i + 1]
        if best_gap > min_gap_ratio:
            header_end = best_boundary

    # --- Footer boundary: largest gap in bottom scan_zone ---
    footer_start = 0.92
    bottom_pos = [p for p in rel_positions if p > (1.0 - scan_zone)]
    if len(bottom_pos) >= 2:
        best_gap, best_boundary = 0.0, footer_start
        for i in range(len(bottom_pos) - 1):
            gap = bottom_pos[i + 1] - bottom_pos[i]
            if gap > best_gap:
                best_gap = gap
                best_boundary = bottom_pos[i]
        if best_gap > min_gap_ratio:
            footer_start = best_boundary

    return header_end, footer_start


def _find_title_font_gap(sizes: list[float]) -> float:
    """Finds the natural gap in the font size distribution.

    Sorts unique sizes, looks for the largest jump between adjacent
    levels in the upper half of the distribution.
    Returns the ratio (level_above_gap / median).
    If no significant gap is found, returns 1.4 (default).
    """
    if not sizes:
        return 1.4

    unique = sorted(set(round(s, 1) for s in sizes))
    if len(unique) < 2:
        return 1.4

    median_size = _median(sizes)
    if median_size == 0:
        return 1.4

    # Only look for gaps in the upper half (above the median)
    upper = [s for s in unique if s >= median_size]
    if len(upper) < 2:
        return 1.4

    best_gap = 0.0
    best_above = upper[-1]
    for i in range(len(upper) - 1):
        gap = upper[i + 1] - upper[i]
        if gap > best_gap:
            best_gap = gap
            best_above = upper[i + 1]

    ratio = best_above / median_size
    # The gap must be significant (at least 15% larger than the median)
    if ratio < 1.15:
        return 1.4

    return round(ratio, 2)


def profile_pdf(pdf_path: str) -> LayoutProfile:
    """Analyzes a PDF and returns a LayoutProfile with numerical metrics."""
    prof = LayoutProfile(pdf_path=pdf_path)

    words_by_page: list[list[dict]] = []
    page_heights: list[float] = []
    all_sizes: list[float] = []
    all_line_gaps: list[float] = []
    all_block_gaps: list[float] = []
    total_words = 0
    multi_col_lines = 0
    total_lines = 0

    with pdfplumber.open(pdf_path) as pdf:
        prof.num_pages = len(pdf.pages)

        for page in pdf.pages:
            words = page.extract_words(
                extra_attrs=["fontname", "size"],
                keep_blank_chars=False,
            )
            if not words:
                words_by_page.append([])
                page_heights.append(page.height)
                continue

            words_by_page.append(words)
            page_heights.append(page.height)
            total_words += len(words)

            sizes = [w.get("size", 10) or 10 for w in words]
            all_sizes.extend(sizes)

            # Group into lines
            sorted_words = sorted(words, key=lambda w: (round(w["top"] / 3.0), w["x0"]))
            lines: list[list[dict]] = []
            current_line = [sorted_words[0]]
            for w in sorted_words[1:]:
                if abs(w["top"] - current_line[-1]["top"]) <= 3.0:
                    current_line.append(w)
                else:
                    lines.append(current_line)
                    current_line = [w]
            lines.append(current_line)

            # Gaps between consecutive lines
            page_line_gaps = []
            for i in range(len(lines) - 1):
                prev_bottom = max(w["bottom"] for w in lines[i])
                curr_top = min(w["top"] for w in lines[i + 1])
                gap = curr_top - prev_bottom
                if gap > 0:
                    page_line_gaps.append(gap)

            all_line_gaps.extend(page_line_gaps)

            # Block gaps: those above median gap * 2
            if page_line_gaps:
                line_gap_threshold = _median(page_line_gaps) * 2
                all_block_gaps.extend(g for g in page_line_gaps if g > line_gap_threshold)

            # Column density
            for line in lines:
                total_lines += 1
                if len(line) >= 2:
                    x0_values = sorted(set(round(w["x0"] / 5) * 5 for w in line))
                    if len(x0_values) >= 3:
                        multi_col_lines += 1

    # --- Aggregate metrics ---
    prof.words_per_page = total_words / max(prof.num_pages, 1)
    prof.font_size_median = _median(all_sizes)
    prof.font_size_std = _std(all_sizes)
    prof.font_size_levels = len(set(round(s, 1) for s in all_sizes))
    prof.line_gap_median = _median(all_line_gaps)
    prof.line_gap_std = _std(all_line_gaps)
    prof.block_gap_median = _median(all_block_gaps) if all_block_gaps else 0.0
    prof.column_density = multi_col_lines / max(total_lines, 1)

    # Boundaries and content density
    header_zone, footer_zone = _find_content_boundaries(words_by_page, page_heights)
    prof.header_zone = round(header_zone, 4)
    prof.footer_zone = round(footer_zone, 4)

    # Words in body zone / total
    body_words = 0
    for words, height in zip(words_by_page, page_heights):
        if height == 0:
            continue
        for w in words:
            rel = w["top"] / height
            if header_zone <= rel <= footer_zone:
                body_words += 1
    prof.content_density = body_words / max(total_words, 1)

    # Natural gap for titles
    prof.title_font_gap = _find_title_font_gap(all_sizes)

    return prof


# ── Parameter derivation (adaptive, per PDF) ───────────────────────────────

def derive_params(prof: LayoutProfile) -> ParsingParams:
    """Computes optimal parsing parameters from measured metrics.

    No lookup tables or hardcoded types: every parameter is a direct
    function of the specific PDF's metrics.
    """
    params = ParsingParams()

    # Header/footer: use detected boundaries with safety margin
    params.header_threshold = max(prof.header_zone, 0.05)
    params.footer_threshold = min(prof.footer_zone, 0.95)

    # Titles: use the natural gap found in the font distribution
    params.title_font_ratio = prof.title_font_gap

    # Line gap: adapt if the document has very tight spacing
    if prof.line_gap_median > 0:
        params.line_gap = min(prof.line_gap_median, 5.0)
    else:
        params.line_gap = 3.0

    # Block gap: calibrate multiplier and minimum on observed spacing
    if prof.line_gap_median > 0 and prof.block_gap_median > 0:
        # The ideal multiplier is what separates lines from blocks
        observed_ratio = prof.block_gap_median / prof.line_gap_median
        params.block_gap_multiplier = max(min(observed_ratio, 4.0), 1.5)
        params.block_gap_minimum = max(prof.line_gap_median * 1.5, 5.0)
    else:
        params.block_gap_multiplier = 2.0
        params.block_gap_minimum = 10.0

    # Columns: if column density is high, the document is tabular
    if prof.column_density > 0.5:
        params.min_columns_for_table = 3
    else:
        params.min_columns_for_table = 4  # more conservative

    # Overlap: dense documents benefit from more context
    if prof.words_per_page > 300:
        params.chunk_overlap = 1
    else:
        params.chunk_overlap = 0

    return params


def profile_and_params(pdf_path: str) -> tuple[LayoutProfile, ParsingParams]:
    """Shortcut: profiles the PDF and returns profile + derived parameters."""
    prof = profile_pdf(pdf_path)
    params = derive_params(prof)
    return prof, params


# ── Persistence and clustering ──────────────────────────────────────────────

class ProfileStore:
    """Saves profiles to disk and clusters them to discover document types.

    Profiles are saved in a JSON file in the database directory.
    Clustering uses normalized Euclidean distance and DBSCAN.
    """

    def __init__(self, db_path: str = "./chroma_db"):
        self.store_path = os.path.join(db_path, "profiles.json")
        self.profiles: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                self.profiles = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def add(self, prof: LayoutProfile):
        """Adds or updates the profile for a PDF."""
        key = os.path.abspath(prof.pdf_path)
        data = asdict(prof)
        self.profiles[key] = data
        self._save()

    def remove(self, pdf_path: str):
        """Removes the profile for a PDF."""
        key = os.path.abspath(pdf_path)
        self.profiles.pop(key, None)
        self._save()

    def list_profiles(self) -> list[LayoutProfile]:
        """Returns all saved profiles."""
        result = []
        for data in self.profiles.values():
            prof = LayoutProfile(**{
                k: v for k, v in data.items()
                if k in LayoutProfile.__dataclass_fields__
            })
            result.append(prof)
        return result

    @staticmethod
    def _auto_eps(X_norm: np.ndarray, percentile: float = 50.0) -> float:
        """Estimates eps as a percentile of pairwise distances.

        With few documents the 50th percentile separates distinct document
        types without fragmenting similar ones.
        """
        from itertools import combinations
        n = len(X_norm)
        if n < 2:
            return 0.5
        dists = [
            float(np.linalg.norm(X_norm[i] - X_norm[j]))
            for i, j in combinations(range(n), 2)
        ]
        return float(np.percentile(dists, percentile))

    def cluster(self, eps: float | None = None, min_samples: int = 1
                ) -> list[list[LayoutProfile]]:
        """Clusters profiles with DBSCAN on normalized features.

        Returns a list of clusters, where each cluster is a list of
        LayoutProfile. Noisy profiles (label -1) each become a
        singleton cluster.

        eps: maximum neighborhood radius (on normalized 0-1 features).
             If None, computed automatically from the 50th percentile
             of pairwise distances.
        min_samples: minimum profiles to form a cluster.
        """
        profiles = self.list_profiles()
        if len(profiles) < 2:
            return [profiles] if profiles else []

        # Feature matrix
        X = np.array([p.feature_vector() for p in profiles])

        # Normalize 0-1 per feature
        mins = X.min(axis=0)
        ranges = X.max(axis=0) - mins
        ranges[ranges == 0] = 1.0  # avoid division by zero
        X_norm = (X - mins) / ranges

        if eps is None:
            eps = self._auto_eps(X_norm)

        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_norm)

        clusters: dict[int, list[LayoutProfile]] = {}
        for label, prof in zip(labels, profiles):
            clusters.setdefault(int(label), []).append(prof)

        # Ordered clusters: real clusters first (label >= 0), then noise
        result = []
        for label in sorted(clusters):
            if label >= 0:
                result.append(clusters[label])
        # Noise: each profile as a singleton
        for prof in clusters.get(-1, []):
            result.append([prof])

        return result
