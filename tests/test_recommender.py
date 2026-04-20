import pytest
from src.recommender import Song, UserProfile, score_song, get_recommendations

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_songs():
    return [
        Song(1, "Happy Pop",     "Artist A", "pop",        "happy",    0.80, 120, 0.85, 0.80, 0.15),
        Song(2, "Chill Lofi",    "Artist B", "lofi",       "chill",    0.35,  75, 0.55, 0.60, 0.82),
        Song(3, "Rock Storm",    "Artist C", "rock",       "intense",  0.90, 150, 0.45, 0.65, 0.10),
        Song(4, "Ambient Float", "Artist D", "ambient",    "chill",    0.25,  60, 0.65, 0.40, 0.91),
        Song(5, "Jazz Cafe",     "Artist E", "jazz",       "relaxed",  0.37,  90, 0.72, 0.55, 0.88),
    ]

@pytest.fixture
def pop_profile():
    return UserProfile("Tester", "pop", "happy", 0.80)

# ── score_song tests ──────────────────────────────────────────────────────────

def test_genre_match_adds_two_points(sample_songs, pop_profile):
    score, reasons = score_song(sample_songs[0], pop_profile)
    assert score >= 2.0
    assert any("Genre match" in r for r in reasons)


def test_non_matching_genre_lower_than_match(sample_songs, pop_profile):
    pop_score, _ = score_song(sample_songs[0], pop_profile)
    rock_score, _ = score_song(sample_songs[2], pop_profile)
    assert pop_score > rock_score


def test_mood_match_adds_points(sample_songs, pop_profile):
    score, reasons = score_song(sample_songs[0], pop_profile)
    assert any("Mood match" in r for r in reasons)
    # genre + mood + near-perfect energy should exceed 4.0
    assert score > 4.0


def test_acoustic_bonus_applied_when_flag_set(sample_songs):
    acoustic_profile = UserProfile("A", "lofi", "chill", 0.35, likes_acoustic=True)
    score, reasons = score_song(sample_songs[1], acoustic_profile)  # acousticness=0.82
    assert any("Acoustic" in r for r in reasons)


def test_no_acoustic_bonus_without_flag(sample_songs):
    non_acoustic = UserProfile("B", "lofi", "chill", 0.35, likes_acoustic=False)
    _, reasons = score_song(sample_songs[1], non_acoustic)
    assert not any("Acoustic" in r for r in reasons)


def test_energy_fit_in_reasons(sample_songs, pop_profile):
    _, reasons = score_song(sample_songs[0], pop_profile)
    assert any("Energy fit" in r for r in reasons)

# ── get_recommendations tests ─────────────────────────────────────────────────

def test_returns_k_results(sample_songs, pop_profile):
    recs = get_recommendations(sample_songs, pop_profile, k=3)
    assert len(recs) == 3


def test_results_sorted_descending(sample_songs, pop_profile):
    recs = get_recommendations(sample_songs, pop_profile, k=5)
    scores = [score for _, score, _ in recs]
    assert scores == sorted(scores, reverse=True)


def test_top_result_matches_genre(sample_songs, pop_profile):
    recs = get_recommendations(sample_songs, pop_profile, k=5)
    top_song, _, _ = recs[0]
    assert top_song.genre == "pop"

# ── guardrail / validation tests ──────────────────────────────────────────────

def test_invalid_energy_raises():
    bad = UserProfile("X", "pop", "happy", 1.5)
    with pytest.raises(ValueError, match="target_energy"):
        get_recommendations([], bad)


def test_empty_genre_raises():
    bad = UserProfile("X", "", "happy", 0.5)
    with pytest.raises(ValueError, match="favorite_genre"):
        get_recommendations([], bad)


def test_empty_mood_raises():
    bad = UserProfile("X", "pop", "", 0.5)
    with pytest.raises(ValueError, match="favorite_mood"):
        get_recommendations([], bad)
