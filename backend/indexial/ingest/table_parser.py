"""
Markdown Table Parser & Database Ingestion

This module:
1. Detects and extracts tables from markdown files
2. Merges multi-page tables using the "Stitcher" algorithm
3. Generates dynamic schemas via Groq LLM
4. Populates tables in Supabase PostgreSQL
5. Maintains a registry of all extracted tables
"""

import re
import json
import uuid
import logging
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from psycopg2 import sql

from indexial.core import config
from indexial.core.db import get_connection
from indexial.ingest.extractor import (
    PAGE_MARKER_RE,
    page_for_offset,
    page_spans,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables


# =================== Data Classes =================== #

@dataclass
class Block:
    """Represents a block of content (TABLE or TEXT)."""
    block_type: str  # "TABLE" or "TEXT"
    lines: List[str] = field(default_factory=list)
    
    def get_content(self) -> str:
        return "\n".join(self.lines)
    
    def get_column_count(self) -> int:
        """Count columns based on pipe characters in first row."""
        if not self.lines:
            return 0
        # Count pipes, subtract 1 for leading pipe (|col1|col2| has 3 pipes, 2 cols)
        first_row = self.lines[0].strip()
        if first_row.startswith("|") and first_row.endswith("|"):
            return first_row.count("|") - 1
        return first_row.count("|") + 1


@dataclass
class ExtractedTable:
    """
    An extracted table, carrying the provenance needed to cite it.

    row_pages runs parallel to rows: each entry is the page that row came
    from. It has to be per-row rather than per-table, because a table stitched
    across a page break has rows from more than one page and a citation that
    points at the wrong page is worse than no citation at all.
    """

    headers: List[str]
    rows: List[List[str]]
    source_file: str
    table_index: int
    raw_markdown: str
    row_pages: List[int] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    # The exact source spans this table was built from: one per page for a
    # stitched table. Held as a list rather than re-split out of raw_markdown,
    # because the join character is not recoverable once concatenated.
    raw_fragments: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Keep row_pages aligned with rows even when pages are unknown: the
        # disk-markdown path has no page information at all.
        if not self.row_pages:
            self.row_pages = [self.page_start or 0] * len(self.rows)
        elif len(self.row_pages) < len(self.rows):
            pad = self.row_pages[-1] if self.row_pages else (self.page_start or 0)
            self.row_pages += [pad] * (len(self.rows) - len(self.row_pages))

        if not self.raw_fragments and self.raw_markdown:
            self.raw_fragments = [self.raw_markdown]

    def column_count(self) -> int:
        return len(self.headers)

    def page_of(self, row_index: int) -> Optional[int]:
        """Page for one row, or None when pages were never recovered."""
        if 0 <= row_index < len(self.row_pages):
            return self.row_pages[row_index] or None
        return self.page_start


# =================== Markdown Table Stitcher =================== #

class MarkdownTableStitcher:
    """
    Implements the "Stitcher" algorithm to detect and merge
    multi-page tables in markdown files.
    """
    
    # Patterns to identify page breaks/markers
    PAGE_BREAK_PATTERNS = [
        r'<!--\s*PAGE\s*\d+\s*-->',  # <!-- PAGE 1 -->
        r'---+',                       # Horizontal rules
        r'^\s*\d+\s*$',               # Just a page number
    ]
    
    def __init__(self):
        self.page_break_regex = re.compile(
            '|'.join(self.PAGE_BREAK_PATTERNS),
            re.IGNORECASE
        )
    
    def process(self, markdown_content: str) -> str:
        """
        Main entry point: Process markdown and return cleaned version
        with merged tables.
        """
        # Phase 1: Block Tokenization
        blocks = self._tokenize_blocks(markdown_content)
        logger.info(f"Phase 1: Tokenized into {len(blocks)} blocks")
        
        # Phase 2: Merge Logic (The "Zipper")
        merged_blocks = self._merge_tables(blocks)
        logger.info(f"Phase 2: After merging, {len(merged_blocks)} blocks remain")
        
        # Phase 4: Reconstruction
        output = self._reconstruct(merged_blocks)
        
        return output
    
    def _tokenize_blocks(self, content: str) -> List[Block]:
        """Phase 1: Break file into discrete TABLE vs TEXT blocks."""
        blocks: List[Block] = []
        lines = content.split("\n")
        
        current_block: Optional[Block] = None
        
        for line in lines:
            is_table_row = line.strip().startswith("|")
            
            if is_table_row:
                # It's a table row
                if current_block is None or current_block.block_type != "TABLE":
                    # Close previous block if exists
                    if current_block is not None:
                        blocks.append(current_block)
                    # Start new TABLE block
                    current_block = Block(block_type="TABLE")
                current_block.lines.append(line)
            else:
                # It's text/empty
                if current_block is None or current_block.block_type != "TEXT":
                    # Close previous block if exists
                    if current_block is not None:
                        blocks.append(current_block)
                    # Start new TEXT block
                    current_block = Block(block_type="TEXT")
                current_block.lines.append(line)
        
        # Don't forget the last block
        if current_block is not None:
            blocks.append(current_block)
        
        return blocks
    
    def _merge_tables(self, blocks: List[Block]) -> List[Block]:
        """Phase 2: Iterate and merge tables if conditions are met."""
        i = 0
        
        while i < len(blocks):
            current_block = blocks[i]
            
            # Check 1: Is current block a table?
            if current_block.block_type != "TABLE":
                i += 1
                continue
            
            # Look ahead: Need at least 2 more blocks (gap + next table)
            if i + 2 >= len(blocks):
                i += 1
                continue
            
            gap_block = blocks[i + 1]
            next_block = blocks[i + 2]
            
            # Check 2: Is the gap empty/whitespace/page markers only?
            if not self._is_empty_gap(gap_block):
                i += 1
                continue
            
            # Check 3: Is next block a table?
            if next_block.block_type != "TABLE":
                i += 1
                continue
            
            # Check 4: Do column counts match?
            cols_a = current_block.get_column_count()
            cols_b = next_block.get_column_count()
            
            if cols_a != cols_b:
                i += 1
                continue
            
            # ACTION: MERGE DETECTED!
            logger.info(f"Merging tables at block {i} (cols={cols_a})")
            
            # Phase 3: Sanitize rows from next table (remove repeated headers)
            sanitized_rows = self._sanitize_rows(current_block, next_block)
            
            # Append sanitized rows to current block
            current_block.lines.extend(sanitized_rows)
            
            # Delete the gap and old next table
            del blocks[i + 1]  # Remove gap
            del blocks[i + 1]  # Remove next table (now at i+1 after gap removal)
            
            # Do NOT increment i - re-check for 3+ page tables
        
        return blocks
    
    def _is_empty_gap(self, block: Block) -> bool:
        """Check if a text block contains only whitespace or page markers."""
        content = block.get_content().strip()
        
        # Empty content
        if not content:
            return True
        
        # Remove all page break patterns
        cleaned = self.page_break_regex.sub("", content).strip()
        
        # Check if anything meaningful remains
        return len(cleaned) == 0
    
    def _sanitize_rows(self, table_a: Block, table_b: Block) -> List[str]:
        """Phase 3: Remove repeated headers from table B."""
        if not table_b.lines:
            return []
        
        rows_b = table_b.lines.copy()
        header_a = table_a.lines[0].strip() if table_a.lines else ""
        
        rows_to_return = []
        
        for idx, row in enumerate(rows_b):
            row_stripped = row.strip()
            
            # Check if it's a header row (matches table A's header)
            if idx == 0 and self._is_similar_header(header_a, row_stripped):
                logger.debug(f"Removing repeated header: {row_stripped[:50]}...")
                continue
            
            # Check if it's a separator row (|---|---|)
            if self._is_separator_row(row_stripped):
                logger.debug(f"Removing separator row: {row_stripped[:50]}...")
                continue
            
            rows_to_return.append(row)
        
        return rows_to_return
    
    def _is_similar_header(self, header_a: str, header_b: str) -> bool:
        """Check if two headers are similar (exact or normalized match)."""
        # Normalize: remove extra spaces, lowercase
        norm_a = re.sub(r'\s+', ' ', header_a.lower().strip())
        norm_b = re.sub(r'\s+', ' ', header_b.lower().strip())
        return norm_a == norm_b
    
    def _is_separator_row(self, row: str) -> bool:
        """Check if row is a markdown table separator (|---|---|)."""
        # Remove pipes and check if only dashes, colons, spaces remain
        cleaned = row.replace("|", "").replace("-", "").replace(":", "").strip()
        return len(cleaned) == 0 and "-" in row
    
    def _reconstruct(self, blocks: List[Block]) -> str:
        """Phase 4: Reconstruct markdown from blocks."""
        parts = []
        for block in blocks:
            parts.append(block.get_content())
        return "\n".join(parts)


# =================== Table Extractor =================== #

class MarkdownTableExtractor:
    """Extracts structured table data from processed markdown."""

    TABLE_PATTERN = re.compile(
        r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)',
        re.MULTILINE,
    )

    def __init__(self):
        self.stitcher = MarkdownTableStitcher()

    # ---------------------------------------------------------- page-aware --

    def extract_tables_from_pages(self, pages, source_file: str) -> List[ExtractedTable]:
        """
        Extract tables page by page, then stitch at the object level.

        This replaces stitching the concatenated markdown as a string. That
        approach destroyed page information twice over: the page separator was
        an anonymous '---', and the stitcher deliberately deleted it. Running
        the extractor per page means every table knows its page before any
        merging happens, and merging two ExtractedTables is then just list
        concatenation.

        Args:
            pages: list of extractor.Page (number, markdown)
            source_file: source filename for metadata

        Returns:
            Stitched ExtractedTable objects, re-indexed in document order.
        """
        found: List[ExtractedTable] = []

        for page in pages:
            for match in self.TABLE_PATTERN.finditer(page.markdown):
                raw_table = match.group(1)
                parsed = self._parse_table(raw_table)
                if not parsed:
                    continue
                headers, rows = parsed
                found.append(
                    ExtractedTable(
                        headers=headers,
                        rows=rows,
                        source_file=source_file,
                        table_index=len(found),
                        raw_markdown=raw_table,
                        row_pages=[page.number] * len(rows),
                        page_start=page.number,
                        page_end=page.number,
                    )
                )

        stitched = self._stitch_tables(found)
        for index, table in enumerate(stitched):
            table.table_index = index

        logger.info(
            f"Extracted {len(stitched)} tables from {source_file} "
            f"({len(found)} fragments across {len(pages)} pages)"
        )
        return stitched

    def _stitch_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """
        Merge a table continuing onto the next page into its predecessor.

        A continuation is recognised when the column count matches and either
        the header repeats verbatim (in which case the repeated header row is
        really data and gets dropped) or the fragment starts on the page
        immediately after.
        """
        if not tables:
            return []

        merged: List[ExtractedTable] = [tables[0]]

        for table in tables[1:]:
            previous = merged[-1]
            contiguous = (table.page_start or 0) - (previous.page_end or 0) in (0, 1)
            same_width = table.column_count() == previous.column_count() and table.column_count() > 0

            if not (contiguous and same_width):
                merged.append(table)
                continue

            headers_repeat = all(
                self.stitcher._is_similar_header(a, b)
                for a, b in zip(previous.headers, table.headers)
            )
            if not headers_repeat:
                merged.append(table)
                continue

            previous.rows.extend(table.rows)
            previous.row_pages.extend(table.row_pages)
            previous.page_end = table.page_end
            previous.raw_markdown = f"{previous.raw_markdown}\n{table.raw_markdown}"
            previous.raw_fragments.extend(table.raw_fragments)

        return merged

    # ------------------------------------------------------- legacy string --

    def extract_tables(self, markdown_content: str, source_file: str) -> List[ExtractedTable]:
        """
        Extract all tables from markdown content.
        
        Args:
            markdown_content: Raw markdown string
            source_file: Source filename for metadata
        
        Returns:
            List of ExtractedTable objects
        """
        # Page markers, if any, must be mapped BEFORE stitching: stitching
        # removes them and shifts every offset after them.
        spans = page_spans(markdown_content)
        has_pages = bool(PAGE_MARKER_RE.search(markdown_content))

        page_by_table: Dict[str, int] = {}
        if has_pages:
            for match in self.TABLE_PATTERN.finditer(markdown_content):
                page_by_table[match.group(1)] = page_for_offset(spans, match.start())

        # Stitch multi-page tables
        cleaned_content = self.stitcher.process(markdown_content)

        tables = []
        for idx, match in enumerate(self.TABLE_PATTERN.finditer(cleaned_content)):
            raw_table = match.group(1)
            parsed = self._parse_table(raw_table)

            if parsed:
                headers, rows = parsed
                page = page_by_table.get(raw_table) if has_pages else None
                tables.append(ExtractedTable(
                    headers=headers,
                    rows=rows,
                    source_file=source_file,
                    table_index=idx,
                    raw_markdown=raw_table,
                    page_start=page,
                    page_end=page,
                ))

        logger.info(f"Extracted {len(tables)} tables from {source_file}")
        return tables
    
    def _parse_table(self, raw_table: str) -> Optional[Tuple[List[str], List[List[str]]]]:
        """Parse raw markdown table into headers and rows."""
        lines = [l.strip() for l in raw_table.strip().split("\n") if l.strip()]
        
        if len(lines) < 2:
            return None
        
        # First line is headers
        headers = self._parse_row(lines[0])
        
        # Find separator line and skip it
        data_start = 1
        for i, line in enumerate(lines[1:], 1):
            if self._is_separator(line):
                data_start = i + 1
                break
        
        # Remaining lines are data rows.
        #
        # Rows whose width does not match the header used to be dropped
        # outright. In a long-format store there is no reason to: a short row
        # simply yields fewer cells, and fact_store flags it as 'ragged_row'.
        # Over-wide rows are truncated, since the extra cells have no header
        # to be addressed by.
        rows = []
        for line in lines[data_start:]:
            row = self._parse_row(line)
            if not row or not any(cell.strip() for cell in row):
                continue
            rows.append(row[: len(headers)] if len(row) > len(headers) else row)

        if not headers or not rows:
            return None

        return headers, rows
    
    def _parse_row(self, row: str) -> List[str]:
        """Parse a single markdown table row into cells."""
        # Remove leading/trailing pipes and split
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        
        cells = [cell.strip() for cell in row.split("|")]
        return cells
    
    def _is_separator(self, line: str) -> bool:
        """Check if line is a separator row."""
        cleaned = line.replace("|", "").replace("-", "").replace(":", "").strip()
        return len(cleaned) == 0 and "-" in line

# The database ingestion path that used to live here is gone.
#
# It asked Groq for a CREATE TABLE statement per extracted table and then
# executed that string unparameterised. Table rows now go to
# ingest/fact_store.py, which writes into one fixed schema with bound
# parameters, so no model-authored DDL reaches the database at all.


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else str(config.MARKDOWN_DIR / "trial.md")
    text = pathlib.Path(path).read_text(encoding="utf-8")
    extracted = MarkdownTableExtractor().extract_tables(text, pathlib.Path(path).name)
    print(f"{len(extracted)} table(s) in {path}")
    for t in extracted:
        pages = f" pages {t.page_start}-{t.page_end}" if t.page_start else ""
        print(f"  [{t.table_index}] {len(t.rows)} rows x {t.column_count()} cols{pages}")
        print(f"      {', '.join(t.headers)}")
