import os
import pytest
from src.retriever import SongDoc, load_descriptions, retrieve, _tokenize

# ── in-memory fixture (no file I/O) ───────────────────────────────────────────

@pytest.fixture
def sample_docs():
    return [
        SongDoc(1,  "Sunrise City",       "Neon Echo",       "upbeat pop morning energy happy danceable feel-good optimistic"),
        SongDoc(2,  "Library Rain",        "Paper Lanterns",  "lofi study focus chill relaxing quiet rain homework concentration"),
        SongDoc(3,  "Storm Runner",        "Voltline",        "rock intense gym workout powerful energy aggressive lifting weights"),
        SongDoc(4,  "Spacewalk Thoughts",  "Orbit Bloom",     "ambient peaceful meditation sleep relaxing calm gentle quiet floating"),
        SongDoc(5,  "Coffee Shop Stories", "Slow Stereo",     "jazz relaxed coffee background reading chill smooth acoustic morning"),
        SongDoc(6,  "Club Ignite",         "DJ Velocity",     "electronic dance party rave gym energetic maximum energy intense hype"),
        SongDoc(7,  "Pillow Clouds",       "Dream Drift",     "lofi sleep bedtime gentle quiet peaceful dreamy calm low energy"),
    ]


# ── tokenizer tests ───────────────────────────────────────────────────────────

def test_tokenize_lowercases_input():
    tokens = _tokenize("Study MUSIC Focus")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_removes_stopwords():
    tokens = _tokenize("the best and a for music")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "for" not in tokens


def test_tokenize_removes_short_tokens():
    tokens = _tokenize("go to gym")
    assert "go" not in tokens  # len 2 — filtered out
    assert "to" not in tokens


# ── retrieve() basic behavior ─────────────────────────────────────────────────

def test_retrieve_returns_k_results(sample_docs):
    results = retrieve(sample_docs, "chill music for studying", k=3)
    assert len(results) == 3


def test_retrieve_returns_fewer_when_catalog_smaller(sample_docs):
    results = retrieve(sample_docs[:2], "any music", k=5)
    assert len(results) == 2


def test_retrieve_empty_docs_returns_empty():
    assert retrieve([], "any query", k=3) == []


def test_retrieve_empty_query_no_crash(sample_docs):
    results = retrieve(sample_docs, "", k=3)
    assert len(results) == 3


# ── relevance tests ───────────────────────────────────────────────────────────

def test_retrieve_study_query_prefers_lofi(sample_docs):
    results = retrieve(sample_docs, "I want chill music for studying and focus", k=2)
    titles = [d.title for d in results]
    assert "Library Rain" in titles


def test_retrieve_gym_query_prefers_high_energy(sample_docs):
    results = retrieve(sample_docs, "high energy gym workout music", k=2)
    titles = [d.title for d in results]
    assert any(t in titles for t in ("Storm Runner", "Club Ignite"))


def test_retrieve_sleep_query_prefers_calm(sample_docs):
    results = retrieve(sample_docs, "something quiet for bedtime sleep", k=2)
    titles = [d.title for d in results]
    assert any(t in titles for t in ("Pillow Clouds", "Spacewalk Thoughts"))


def test_retrieve_results_are_sorted_by_relevance(sample_docs):
    """Calling twice with the same query should produce the same ordered results."""
    r1 = retrieve(sample_docs, "intense workout rock gym", k=3)
    r2 = retrieve(sample_docs, "intense workout rock gym", k=3)
    assert [d.song_id for d in r1] == [d.song_id for d in r2]


# ── load_descriptions integration test ───────────────────────────────────────

def test_load_descriptions_count():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "song_descriptions.txt")
    docs = load_descriptions(path)
    assert len(docs) == 20


def test_load_descriptions_fields():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "song_descriptions.txt")
    docs = load_descriptions(path)
    for doc in docs:
        assert doc.song_id > 0
        assert doc.title != ""
        assert doc.artist != ""
        assert len(doc.description) > 20


def test_load_descriptions_ids_are_unique():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "song_descriptions.txt")
    docs = load_descriptions(path)
    ids = [d.song_id for d in docs]
    assert len(ids) == len(set(ids))
