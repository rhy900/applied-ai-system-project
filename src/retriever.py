"""
RAG Retriever — keyword-based TF-IDF document retrieval for song descriptions.

Loads song_descriptions.txt, scores each document against a user query using
term-frequency * inverse-document-frequency weighting, and returns the top-k
most relevant SongDocs to inject into the Gemini CHECK prompt.
"""

import re
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "it", "for", "and", "or", "to", "of",
    "in", "with", "that", "this", "are", "be", "by", "its", "if",
    "on", "at", "as", "you", "your", "from", "but", "not", "any",
    "has", "have", "when", "can", "will", "more", "like", "so",
    "all", "one", "they", "their", "them", "than", "into",
}


@dataclass
class SongDoc:
    song_id: int
    title: str
    artist: str
    description: str


def load_descriptions(path: str) -> list[SongDoc]:
    """Parse song_descriptions.txt and return one SongDoc per song."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # split on [N] markers — entries alternate: ['', '1', '\nTITLE:...\n', '2', ...]
    parts = re.split(r'\[(\d+)\]', content)
    docs: list[SongDoc] = []

    for i in range(1, len(parts), 2):
        song_id = int(parts[i])
        block = parts[i + 1].strip()
        lines = block.splitlines()

        title = artist = ""
        desc_lines: list[str] = []
        for line in lines:
            if line.startswith("TITLE:"):
                title = line.partition(":")[2].strip()
            elif line.startswith("ARTIST:"):
                artist = line.partition(":")[2].strip()
            elif line.strip():
                desc_lines.append(line.strip())

        docs.append(SongDoc(
            song_id=song_id,
            title=title,
            artist=artist,
            description=" ".join(desc_lines),
        ))

    logger.info(f"Loaded {len(docs)} song descriptions from {path}")
    return docs


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _build_idf(docs: list[SongDoc]) -> dict[str, float]:
    """Compute IDF for every term across the corpus."""
    n = len(docs)
    doc_freq: dict[str, int] = {}
    for doc in docs:
        tokens = set(_tokenize(doc.description + " " + doc.title))
        for t in tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    # smoothed IDF: log((N+1)/(df+1)) + 1
    return {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in doc_freq.items()}


def retrieve(docs: list[SongDoc], query: str, k: int = 3) -> list[SongDoc]:
    """
    Return the top-k SongDocs most relevant to the query using TF-IDF scoring.
    Falls back to the first k docs if query produces no tokens.
    """
    if not docs:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        logger.debug("Empty query tokens — returning first %d docs", k)
        return docs[:k]

    idf = _build_idf(docs)
    scored: list[tuple[SongDoc, float]] = []

    for doc in docs:
        full_text = f"{doc.description} {doc.title} {doc.artist}"
        doc_tokens = _tokenize(full_text)
        tf: dict[str, float] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        if doc_tokens:
            for t in tf:
                tf[t] /= len(doc_tokens)

        score = sum(tf.get(t, 0.0) * idf.get(t, 1.0) for t in query_tokens)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [doc for doc, _ in scored[:k]]
    logger.debug("RAG retrieved: %s", [d.title for d in top])
    return top
