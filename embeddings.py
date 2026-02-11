"""
Shared Jina Embeddings Module

Single canonical JinaEmbeddingClient used by all modules.
Based on the batch-capable version from chunker.py.
"""

import os
import logging
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class JinaEmbeddingClient:
    """
    Jina AI embeddings client for generating vectors.
    Supports single and batch embeddings for efficiency.
    """

    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required")

        self.api_url = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
        self.model = os.getenv("JINA_MODEL", "jina-embeddings-v3")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self.task = os.getenv("JINA_TASK", "text-matching")
        self.batch_size = int(os.getenv("JINA_BATCH_SIZE", "32"))

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Initialized JinaEmbeddingClient with model={self.model}, "
            f"dimensions={self.dimensions}"
        )

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        embeddings = self.embed_batch([text])
        return embeddings[0] if embeddings else []

    def embed_query(self, text: str) -> List[float]:
        """Alias for embed() for compatibility."""
        return self.embed(text)

    def embed_batch(self, texts: List[str], max_chars: int = 12000) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts in batched API calls.

        Args:
            texts: List of texts to embed
            max_chars: Maximum characters per text (truncated if exceeded)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        truncated_texts = [text[:max_chars] for text in texts]
        all_embeddings = []

        for i in range(0, len(truncated_texts), self.batch_size):
            batch = truncated_texts[i : i + self.batch_size]

            payload = {
                "model": self.model,
                "task": self.task,
                "dimensions": self.dimensions,
                "input": batch,
            }

            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Embedded batch {i // self.batch_size + 1}, {len(batch)} texts"
                )

            except requests.exceptions.RequestException as e:
                logger.error(f"Jina API request failed: {e}")
                all_embeddings.extend([[] for _ in batch])
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected response format from Jina API: {e}")
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings
