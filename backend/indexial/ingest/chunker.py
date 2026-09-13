"""
Semantic Chunking Module

This module provides semantic chunking functionality using the semchunk library
to split text into semantically meaningful chunks, with embedding generation
and Supabase pgvector storage capabilities.
"""

import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any

import semchunk
from psycopg2.extras import execute_values

from indexial.core import config
from indexial.core.db import get_connection
from indexial.providers.embeddings import JinaEmbeddingClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables


class SemanticChunker:
    """
    A class for semantically chunking text using the semchunk library.
    
    Attributes:
        chunk_size: Maximum number of tokens per chunk
        overlap: Overlap ratio or token count between chunks
        tokenizer: Tokenizer name or custom token counter
        chunker: The semchunk chunker instance
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: Optional[float | int] = None,
        tokenizer: str | Callable[[str], int] = "gpt-4",
        memoize: bool = True,
    ):
        """
        Initialize the SemanticChunker.
        
        Args:
            chunk_size: Maximum number of tokens a chunk may contain.
            overlap: Proportion (< 1) or absolute number of tokens (>= 1) 
                     by which chunks should overlap. None for no overlap.
            tokenizer: Name of a tiktoken/transformers tokenizer, or a custom
                       token counting function.
            memoize: Whether to memoize the token counter for performance.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer
        self.memoize = memoize
        
        # Create the chunker using semchunk.chunkerify()
        self.chunker = semchunk.chunkerify(
            tokenizer_or_token_counter=tokenizer,
            chunk_size=chunk_size,
            memoize=memoize,
        )
        
        logger.info(
            f"Initialized SemanticChunker with chunk_size={chunk_size}, "
            f"overlap={overlap}, tokenizer={tokenizer}"
        )
    
    def chunk_text(
        self,
        text: str,
        offsets: bool = False,
    ) -> list[str] | tuple[list[str], list[tuple[int, int]]]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: The text to be chunked.
            offsets: If True, return start and end offsets of each chunk.
        
        Returns:
            A list of chunks, or a tuple of (chunks, offsets) if offsets=True.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return ([], []) if offsets else []
        
        result = self.chunker(text, offsets=offsets, overlap=self.overlap)
        
        if offsets:
            chunks, chunk_offsets = result
            logger.info(f"Created {len(chunks)} chunks with offsets")
            return chunks, chunk_offsets
        else:
            logger.info(f"Created {len(result)} chunks")
            return result
    
    def chunk_texts(
        self,
        texts: list[str],
        offsets: bool = False,
        processes: int = 1,
        progress: bool = False,
    ) -> list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]:
        """
        Split multiple texts into semantically meaningful chunks.
        
        Args:
            texts: List of texts to be chunked.
            offsets: If True, return start and end offsets of each chunk.
            processes: Number of processes for multiprocessing (> 1 to enable).
            progress: If True, display a progress bar.
        
        Returns:
            A list of lists of chunks, or a tuple with offsets if offsets=True.
        """
        if not texts:
            logger.warning("Empty text list provided for chunking")
            return ([], []) if offsets else []
        
        result = self.chunker(
            texts,
            offsets=offsets,
            overlap=self.overlap,
            processes=processes,
            progress=progress,
        )
        
        if offsets:
            chunks_list, offsets_list = result
            total_chunks = sum(len(c) for c in chunks_list)
            logger.info(f"Created {total_chunks} chunks from {len(texts)} texts")
            return chunks_list, offsets_list
        else:
            total_chunks = sum(len(c) for c in result)
            logger.info(f"Created {total_chunks} chunks from {len(texts)} texts")
            return result


# =================== Supabase Vector Store =================== #

class SupabaseVectorStore:
    """
    Supabase PostgreSQL vector database client for storing and retrieving embeddings.
    Uses pgvector extension for similarity search.
    Uses deterministic IDs for idempotent upserts.
    """
    
    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or config.VECTOR_TABLE_NAME
        self.vector_size = config.EMBEDDING_DIMENSIONS

        self._ensure_table_exists()
        logger.info(f"Initialized SupabaseVectorStore with table={self.table_name}")

    def _ensure_table_exists(self):
        """
        Create the schema if needed.

        Delegates to core.schema so there is one definition of the vector table
        rather than a private copy here. That shared version also builds an
        HNSW index instead of the ivfflat(lists=100) this used to create:
        ivfflat has to be trained against representative data to be useful, and
        this corpus is wiped and rebuilt constantly, so its lists were always
        badly calibrated.
        """
        from indexial.core.schema import ensure_schema

        try:
            ensure_schema(vector_table=self.table_name, dimensions=self.vector_size)
            logger.info(f"Table {self.table_name} verified/created")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    @staticmethod
    def generate_deterministic_id(content: str, source_file: str, chunk_index: int) -> str:
        """
        Generate a deterministic ID based on content hash.
        This allows idempotent upserts - same content always gets same ID.
        """
        unique_string = f"{source_file}:{chunk_index}:{content[:500]}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:32]
    
    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Upsert multiple chunks with their embeddings to Supabase.
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: Corresponding embedding vectors
        
        Returns:
            Number of successfully upserted points
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        data_rows = []
        for chunk, embedding in zip(chunks, embeddings):
            if not embedding:
                logger.warning(f"Skipping chunk {chunk.get('chunk_id')} - empty embedding")
                continue
            
            # Generate deterministic ID
            chunk_id_hash = self.generate_deterministic_id(
                content=chunk["content"],
                source_file=chunk.get("source_file", "unknown"),
                chunk_index=chunk.get("chunk_id", 0)
            )
            
            # Build metadata dict for extra fields
            metadata = {
                "chunk_size_tokens": chunk.get("chunk_size_tokens"),
                "overlap": chunk.get("overlap"),
            }
            
            data_rows.append((
                chunk_id_hash,
                chunk.get("document_id"),
                chunk.get("source_file", ""),
                chunk.get("chunk_id", 0),
                chunk["content"],
                chunk.get("start_offset", 0),
                chunk.get("end_offset", 0),
                chunk.get("heading_context", ""),
                chunk.get("section", ""),
                chunk.get("processed_date", datetime.now().isoformat()),
                embedding,
                json.dumps(metadata)
            ))

        if not data_rows:
            logger.warning("No valid chunks to upsert")
            return 0

        with get_connection() as conn:
            cur = conn.cursor()
            try:
                # Use ON CONFLICT for idempotent upserts
                query = f"""
                    INSERT INTO {self.table_name}
                    (id, document_id, source_file, chunk_id, content, start_offset, end_offset,
                     heading_context, section, processed_date, embedding, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        processed_date = EXCLUDED.processed_date
                """

                execute_values(cur, query, data_rows)
                conn.commit()
                logger.info(f"Upserted {len(data_rows)} chunks to Supabase")
                return len(data_rows)

            except Exception as e:
                conn.rollback()
                logger.error(f"Error upserting to Supabase: {e}")
                return 0
            finally:
                cur.close()
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (optional)
        
        Returns:
            List of matching chunks with scores
        """
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            try:
                # Use cosine distance operator <=>
                # Similarity = 1 - distance
                if score_threshold is not None:
                    sql_query = f"""
                        SELECT id, source_file, chunk_id, content, start_offset, end_offset,
                               heading_context, section, processed_date, metadata,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    cur.execute(sql_query, (query_vector, query_vector, score_threshold, query_vector, limit))
                else:
                    sql_query = f"""
                        SELECT id, source_file, chunk_id, content, start_offset, end_offset,
                               heading_context, section, processed_date, metadata,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM {self.table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    cur.execute(sql_query, (query_vector, query_vector, limit))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "source_file": row[1],
                        "chunk_id": row[2],
                        "content": row[3],
                        "start_offset": row[4],
                        "end_offset": row[5],
                        "heading_context": row[6],
                        "section": row[7],
                        "processed_date": row[8],
                        "metadata": row[9],
                        "score": row[10]
                    })

                return results

            except Exception as e:
                logger.error(f"Error searching Supabase: {e}")
                return []
            finally:
                cur.close()


# =================== Document Processor =================== #

class DocumentProcessor:
    """
    Main processor that combines chunking, embedding, and vector storage.
    Implements batch processing with heading context for better retrieval.
    
    This class is responsible ONLY for:
    - Chunking text into semantic chunks
    - Generating embeddings
    - Storing embeddings in Supabase (pgvector)
    
    For retrieval and answering, use retrieval.py
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: Optional[float | int] = 0.1,
        tokenizer: str = "gpt-4",
        output_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embed_with_context: bool = True,
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Overlap ratio or token count
            tokenizer: Tokenizer name for semchunk
            output_dir: Directory for saving outputs
            collection_name: Supabase vector table name
            embed_with_context: Whether to prepend heading context when embedding
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        self.embed_with_context = embed_with_context
        
        # Initialize components
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
        self.embedder = JinaEmbeddingClient()
        self.vector_store = SupabaseVectorStore(table_name=collection_name)
        
        # Create output directories
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, overlap={overlap}")
    
    def extract_heading_context(self, text: str, start_offset: int) -> str:
        """
        Extract the most recent heading before the chunk position.
        Returns heading context for better semantic embedding.
        """
        # Find all markdown headings before this position
        text_before = text[:start_offset]
        
        # Match headings (# to ####)
        heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(text_before))
        
        if not headings:
            return ""
        
        # Build hierarchy from most recent headings at each level
        hierarchy = {}
        for match in headings:
            level = len(match.group(1))
            title = match.group(2).strip()
            hierarchy[level] = title
            # Clear lower level headings when a higher level is found
            for l in list(hierarchy.keys()):
                if l > level:
                    del hierarchy[l]
        
        # Build context string
        context_parts = [hierarchy.get(i, "") for i in sorted(hierarchy.keys())]
        return " > ".join(filter(None, context_parts))
    
    def process_markdown_file(
        self,
        input_path: str,
        save_chunks: bool = True,
        store_to_qdrant: bool = True,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a markdown file: chunk, embed, and store.

        Args:
            input_path: Path to the markdown file
            save_chunks: Whether to save chunks as JSON files
            store_to_qdrant: Whether to store embeddings in Supabase
            document_id: UUID of the parent document (for cross-doc queries)

        Returns:
            List of processed chunk dictionaries
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Processing: {input_path}")

        # Read the file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk the text with offsets
        chunks, offsets = self.chunker.chunk_text(text, offsets=True)

        # Build chunk data with metadata
        chunk_data = []
        for i, (chunk_content, (start, end)) in enumerate(zip(chunks, offsets)):
            # Extract heading context for this chunk position
            heading_context = self.extract_heading_context(text, start)

            chunk_info = {
                "chunk_id": i,
                "document_id": document_id,
                "content": chunk_content,
                "start_offset": start,
                "end_offset": end,
                "source_file": str(input_path),
                "heading_context": heading_context,
                "section": heading_context.split(" > ")[0] if heading_context else "",
                "chunk_size_tokens": self.chunk_size,
                "overlap": self.overlap,
                "processed_date": datetime.now().isoformat(),
            }
            chunk_data.append(chunk_info)
        
        logger.info(f"Created {len(chunk_data)} chunks from {input_path.name}")
        
        # Save chunks locally
        if save_chunks:
            self._save_chunks(input_path.stem, chunk_data)
        
        # Generate embeddings and store to Qdrant
        if store_to_qdrant:
            self._embed_and_store(chunk_data)
        
        return chunk_data
    
    def _save_chunks(self, base_name: str, chunk_data: List[Dict[str, Any]]):
        """Save chunks to JSON files."""
        # Save individual chunks
        for chunk in chunk_data:
            chunk_file = self.chunks_dir / f"{base_name}_chunk_{chunk['chunk_id']:04d}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        # Save all chunks in a single file
        all_chunks_file = self.chunks_dir / f"{base_name}_all_chunks.json"
        with open(all_chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chunks to {self.chunks_dir}")
    
    def _embed_and_store(self, chunk_data: List[Dict[str, Any]]):
        """Generate embeddings and store to Qdrant with batch processing."""
        # Prepare texts for embedding
        # Improvement: Prepend heading context for better semantic search
        texts_to_embed = []
        for chunk in chunk_data:
            if self.embed_with_context and chunk.get("heading_context"):
                # Prepend context to content for richer embedding
                text = f"{chunk['heading_context']}\n\n{chunk['content']}"
            else:
                text = chunk["content"]
            texts_to_embed.append(text)
        
        logger.info(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        
        # Batch embed all chunks
        embeddings = self.embedder.embed_batch(texts_to_embed)
        
        # Store to Supabase
        success_count = self.vector_store.upsert_chunks(chunk_data, embeddings)
        logger.info(f"Successfully stored {success_count}/{len(chunk_data)} chunks to Supabase")


def chunk_markdown_file(
    input_path: str,
    output_dir: Optional[str] = None,
    chunk_size: int = 512,
    overlap: Optional[float | int] = 0.1,
    tokenizer: str = "gpt-4",
    save_chunks: bool = True,
) -> list[dict]:
    """
    Read a markdown file, chunk it semantically, and optionally save chunks.
    
    Args:
        input_path: Path to the input markdown file.
        output_dir: Directory to save the chunked output.
        chunk_size: Maximum number of tokens per chunk.
        overlap: Overlap ratio or token count between chunks.
        tokenizer: Tokenizer name for token counting.
        save_chunks: Whether to save chunks to JSON files.
    
    Returns:
        List of dictionaries containing chunk data with metadata.
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read the markdown file
    logger.info(f"Reading markdown file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Initialize the chunker
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        tokenizer=tokenizer,
    )
    
    # Chunk the text with offsets
    chunks, offsets = chunker.chunk_text(text, offsets=True)
    
    # Create chunk data with metadata
    chunk_data = []
    for i, (chunk, (start, end)) in enumerate(zip(chunks, offsets)):
        chunk_info = {
            "chunk_id": i,
            "content": chunk,
            "start_offset": start,
            "end_offset": end,
            "source_file": str(input_path),
            "chunk_size_tokens": chunk_size,
            "overlap": overlap,
        }
        chunk_data.append(chunk_info)
    
    logger.info(f"Created {len(chunk_data)} chunks from {input_path.name}")
    
    # Save chunks if requested
    if save_chunks:
        output_dir = Path(output_dir) if output_dir else config.CHUNKS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual chunks
        base_name = input_path.stem
        
        for chunk_info in chunk_data:
            chunk_file = output_dir / f"{base_name}_chunk_{chunk_info['chunk_id']:04d}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk_info, f, indent=2, ensure_ascii=False)
        
        # Save all chunks in a single file
        all_chunks_file = output_dir / f"{base_name}_all_chunks.json"
        with open(all_chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chunks to {output_dir}")
    
    return chunk_data


def main():
    """Main function to chunk and embed trial.md file."""
    
    # Path to the trial.md file
    input_file = str(config.MARKDOWN_DIR / "trial.md")
    output_directory = str(config.OUTPUT_DIR)
    
    # Chunking parameters
    chunk_size = 512  # Maximum tokens per chunk
    overlap = 0.1     # 10% overlap between chunks
    tokenizer = "gpt-4"  # Use GPT-4's tokenizer (cl100k_base encoding)
    
    print("=" * 60)
    print("Semantic Chunking with semchunk + Supabase Storage")
    print("=" * 60)
    print(f"\nInput file: {input_file}")
    print(f"Chunk size: {chunk_size} tokens")
    print(f"Overlap: {overlap * 100}%")
    print(f"Tokenizer: {tokenizer}")
    print()
    
    try:
        # Initialize the document processor
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            overlap=overlap,
            tokenizer=tokenizer,
            output_dir=output_directory,
            embed_with_context=True,  # Prepend heading context for better retrieval
        )
        
        # Process the markdown file
        chunks = processor.process_markdown_file(
            input_path=input_file,
            save_chunks=True,
            store_to_qdrant=True,
        )
        
        print(f"\n✅ Successfully created {len(chunks)} chunks")
        print(f"📁 Chunks saved to: {output_directory}/chunks/")
        print(f"🔍 Embeddings stored in Supabase pgvector table")
        
        # Display summary of first few chunks
        print("\n" + "-" * 60)
        print("Sample Chunks Preview:")
        print("-" * 60)
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[Chunk {chunk['chunk_id']}]")
            print(f"Offset: {chunk['start_offset']} - {chunk['end_offset']}")
            if chunk.get('heading_context'):
                print(f"Context: {chunk['heading_context']}")
            # Show first 200 characters of content
            preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            print(f"Content: {preview}")
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")
        
        print("\n" + "=" * 60)
        print("✅ Chunking and embedding complete!")
        print("Use retrieval.py to search and answer questions.")
        print("=" * 60)
        
        return chunks
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return []
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Make sure JINA_API_KEY and SUPABASE_DB_URL are set.")
        return []
    except Exception as e:
        logger.exception("Error during processing")
        print(f"❌ Error: {e}")
        return []


if __name__ == "__main__":
    main()
