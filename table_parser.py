"""
Markdown Table Parser & Database Ingestion

This module:
1. Detects and extracts tables from markdown files
2. Merges multi-page tables using the "Stitcher" algorithm
3. Generates dynamic schemas via Groq LLM
4. Populates tables in Supabase PostgreSQL
5. Maintains a registry of all extracted tables
"""

import os
import re
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import psycopg2
from psycopg2 import sql
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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
    """Represents an extracted table with headers and rows."""
    headers: List[str]
    rows: List[List[str]]
    source_file: str
    table_index: int
    raw_markdown: str


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
    
    def __init__(self):
        self.stitcher = MarkdownTableStitcher()
    
    def extract_tables(self, markdown_content: str, source_file: str) -> List[ExtractedTable]:
        """
        Extract all tables from markdown content.
        
        Args:
            markdown_content: Raw markdown string
            source_file: Source filename for metadata
        
        Returns:
            List of ExtractedTable objects
        """
        # First, stitch multi-page tables
        cleaned_content = self.stitcher.process(markdown_content)
        
        # Find all tables in cleaned content
        tables = []
        table_pattern = re.compile(
            r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)',
            re.MULTILINE
        )
        
        for idx, match in enumerate(table_pattern.finditer(cleaned_content)):
            raw_table = match.group(1)
            parsed = self._parse_table(raw_table)
            
            if parsed:
                headers, rows = parsed
                tables.append(ExtractedTable(
                    headers=headers,
                    rows=rows,
                    source_file=source_file,
                    table_index=idx,
                    raw_markdown=raw_table
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
        
        # Remaining lines are data rows
        rows = []
        for line in lines[data_start:]:
            row = self._parse_row(line)
            if row and len(row) == len(headers):
                rows.append(row)
        
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


# =================== Groq Schema Generator =================== #

class GroqSchemaGenerator:
    """Uses Groq LLM to generate dynamic SQL schemas."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def generate_schema(
        self,
        headers: List[str],
        sample_rows: List[List[str]],
        table_name: str
    ) -> str:
        """
        Generate CREATE TABLE statement using Groq LLM.
        
        Args:
            headers: Column headers
            sample_rows: Sample data rows for type inference
            table_name: Target table name
        
        Returns:
            CREATE TABLE SQL statement
        """
        prompt = f"""You are a PostgreSQL Expert. Generate a CREATE TABLE statement.

Rules:
1. Table Name: {table_name}
2. Add 'id' SERIAL PRIMARY KEY as the FIRST column.
3. Analyze these headers: {headers}
4. Analyze these sample rows: {sample_rows[:3]}
5. Infer appropriate PostgreSQL types (TEXT, INTEGER, NUMERIC, BOOLEAN, DATE, etc.)
6. Sanitize column names: lowercase, underscores for spaces, remove special characters.
7. Output JSON ONLY with format: {{"sql": "CREATE TABLE..."}}

Example output:
{{"sql": "CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, column_name TEXT, another_col INTEGER);"}}
"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            return result.get("sql", "")
        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            # Fallback: Generate simple TEXT columns
            return self._fallback_schema(headers, table_name)
    
    def _fallback_schema(self, headers: List[str], table_name: str) -> str:
        """Generate fallback schema with all TEXT columns."""
        sanitized = [self._sanitize_column_name(h) for h in headers]
        columns = ["id SERIAL PRIMARY KEY"]
        columns.extend([f"{col} TEXT" for col in sanitized])
        return f"CREATE TABLE {table_name} ({', '.join(columns)});"
    
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for PostgreSQL."""
        # Lowercase, replace spaces/special chars with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
        sanitized = sanitized.strip('_')
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'col_' + sanitized
        return sanitized or 'column'
    
    def generate_semantic_description(
        self,
        headers: List[str],
        sample_rows: List[List[str]],
        max_length: int = 100
    ) -> str:
        """Generate a concise semantic description of the table using Groq LLM.
        
        Args:
            headers: Column headers
            sample_rows: Sample data rows (first 2-3 rows)
            max_length: Maximum characters for description
        
        Returns:
            Short semantic description string
        """
        prompt = f"""Analyze this table and provide a concise 1-sentence description (max {max_length} chars).

        Headers: {headers}
        Sample: {sample_rows[:2]}

        Output JSON: {{"description": "your description here"}}"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
            "response_format": {"type": "json_object"},
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            description = result.get("description", "")[:max_length]
            return description if description else self._fallback_description(headers)
        except Exception as e:
            logger.warning(f"Semantic description generation failed: {e}")
            return self._fallback_description(headers)
    
    def _fallback_description(self, headers: List[str]) -> str:
        """Generate fallback description from headers."""
        if len(headers) <= 3:
            return f"Table with {', '.join(headers)} data"
        return f"Table with {len(headers)} columns including {', '.join(headers[:2])}..."


# =================== Database Ingestion =================== #

class TableIngestionPipeline:
    """Handles database operations for table ingestion."""
    
    def __init__(self):
        self.db_url = os.getenv("SUPABASE_DB_URL")
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL environment variable is required")
        
        self.schema_generator = GroqSchemaGenerator()
        
        # Ensure registry table exists
        self._ensure_registry_exists()
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def _ensure_registry_exists(self):
        """Create the table_registry if it doesn't exist."""
        create_registry_sql = """
        CREATE TABLE IF NOT EXISTS table_registry (
            id SERIAL PRIMARY KEY,
            physical_table_name TEXT UNIQUE NOT NULL,
            source_doc_uuid UUID NOT NULL,
            original_filename TEXT,
            semantic_description TEXT,
            headers JSONB,
            row_count INTEGER,
            extracted_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(create_registry_sql)
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Registry table verified/created")
        except Exception as e:
            logger.error(f"Failed to create registry table: {e}")
            raise
    
    def ingest_table(self, table: ExtractedTable) -> Optional[str]:
        """
        Ingest a single extracted table into the database.
        
        Args:
            table: ExtractedTable object
        
        Returns:
            Physical table name if successful, None otherwise
        """
        if not table.headers or not table.rows:
            logger.warning("Empty table provided for ingestion")
            return None
        
        # Generate unique identifiers
        doc_uuid = str(uuid.uuid4())
        physical_table_name = f"tbl_{doc_uuid.split('-')[0]}_extracted"
        
        logger.info(f"--- Ingesting Table {table.table_index} from {table.source_file} ---")
        logger.info(f"Target Table: {physical_table_name}")
        logger.info(f"Columns: {table.headers}")
        logger.info(f"Rows: {len(table.rows)}")
        
        # 1. Generate schema and semantic description via Groq
        logger.info("Generating schema via Groq...")
        create_table_sql = self.schema_generator.generate_schema(
            table.headers,
            table.rows[:3],
            physical_table_name
        )
        logger.info(f"Generated SQL: {create_table_sql[:100]}...")
        
        logger.info("Generating semantic description...")
        semantic_description = self.schema_generator.generate_semantic_description(
            table.headers,
            table.rows[:3]
        )
        logger.info(f"Description: {semantic_description}")
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            # 2. Create the table
            logger.info("Executing DDL...")
            cur.execute(create_table_sql)
            
            # 3. Insert data
            logger.info(f"Inserting {len(table.rows)} rows...")
            
            # Create placeholders
            placeholders = ",".join(["%s"] * len(table.headers))
            
            # Build insert query
            insert_query = sql.SQL("INSERT INTO {} VALUES (DEFAULT, {})").format(
                sql.Identifier(physical_table_name),
                sql.SQL(placeholders)
            )
            
            cur.executemany(insert_query, table.rows)
            
            # 4. Update registry
            logger.info("Updating Table Registry...")
            cur.execute("""
                INSERT INTO table_registry 
                (physical_table_name, source_doc_uuid, semantic_description, 
                 original_filename, headers, row_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                physical_table_name,
                doc_uuid,
                semantic_description,
                table.source_file,
                json.dumps(table.headers),
                len(table.rows)
            ))
            
            conn.commit()
            logger.info(f"✅ Success! Table '{physical_table_name}' created with {len(table.rows)} rows")
            
            return physical_table_name
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Ingestion failed: {e}")
            return None
        finally:
            cur.close()
            conn.close()
    
    def ingest_all_tables(self, tables: List[ExtractedTable]) -> List[str]:
        """
        Ingest multiple tables.
        
        Args:
            tables: List of ExtractedTable objects
        
        Returns:
            List of successfully created table names
        """
        created_tables = []
        
        for table in tables:
            table_name = self.ingest_table(table)
            if table_name:
                created_tables.append(table_name)
        
        return created_tables


# =================== Main Pipeline =================== #

def process_markdown_file(
    file_path: str,
    ingest_to_db: bool = True,
    dry_run: bool = False
) -> List[ExtractedTable]:
    """
    Main function to process a markdown file.
    
    Args:
        file_path: Path to markdown file
        ingest_to_db: Whether to ingest tables to database
        dry_run: If True, extract tables but don't write to DB
    
    Returns:
        List of extracted tables
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Processing: {file_path}")
    
    # Read markdown content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract tables
    extractor = MarkdownTableExtractor()
    tables = extractor.extract_tables(content, file_path.name)
    
    if not tables:
        logger.info("No tables found in file")
        return []
    
    logger.info(f"Found {len(tables)} tables")
    
    # Preview tables
    for i, table in enumerate(tables):
        print(f"\n--- Table {i} ---")
        print(f"Headers: {table.headers}")
        print(f"Rows: {len(table.rows)}")
        if table.rows:
            print(f"Sample row: {table.rows[0]}")
    
    # Ingest to database
    if ingest_to_db and not dry_run:
        try:
            pipeline = TableIngestionPipeline()
            created = pipeline.ingest_all_tables(tables)
            print(f"\n✅ Created {len(created)} tables in database")
            for name in created:
                print(f"   - {name}")
        except ValueError as e:
            print(f"\n⚠️ Database ingestion skipped: {e}")
    elif dry_run:
        print("\n🔍 Dry run - no database writes performed")
    
    return tables


def main():
    """CLI entry point."""
    import sys
    
    # Default file path
    default_path = "output/markdown/trial.md"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        dry_run = "--dry-run" in sys.argv
    else:
        file_path = default_path
        dry_run = False
    
    print("=" * 60)
    print("Markdown Table Parser & Database Ingestion")
    print("=" * 60)
    print(f"\nFile: {file_path}")
    print(f"Dry Run: {dry_run}")
    print()
    
    try:
        tables = process_markdown_file(file_path, ingest_to_db=True, dry_run=dry_run)
        
        if tables:
            print(f"\n✅ Processed {len(tables)} tables successfully")
        else:
            print("\n⚠️ No tables found in the document")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        logger.exception("Processing failed")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
