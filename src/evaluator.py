"""
Test harness — runs predefined user profiles through the scoring engine
and reports pass/fail without making any API calls.
"""

from dataclasses import dataclass
from src.recommender import Song, UserProfile, get_recommendations


@dataclass
class TestCase:
    profile: UserProfile
    expected_genre: str
    expected_mood: str
    description: str


@dataclass
class TestResult:
    case: TestCase
    passed: bool
    top_title: str
    top_genre: str
    top_mood: str
    top_score: float
    notes: str


TEST_CASES: list[TestCase] = [
    TestCase(UserProfile("Pop Fan",      "pop",        "happy",     0.80),        "pop",        "happy",     "Pop/happy high-energy listener"),
    TestCase(UserProfile("Lofi Lover",   "lofi",       "chill",     0.37, True),  "lofi",       "chill",     "Lofi/chill acoustic listener"),
    TestCase(UserProfile("Rock Head",    "rock",       "intense",   0.91),        "rock",       "intense",   "Rock/intense max-energy fan"),
    TestCase(UserProfile("Ambient Fan",  "ambient",    "chill",     0.25, True),  "ambient",    "chill",     "Ambient/chill acoustic listener"),
    TestCase(UserProfile("Jazz Lover",   "jazz",       "relaxed",   0.37, True),  "jazz",       "relaxed",   "Jazz/relaxed acoustic fan"),
    TestCase(UserProfile("EDM Head",     "electronic", "energetic", 0.97),        "electronic", "energetic", "Electronic/energetic max-energy fan"),
]


def run_test(songs: list[Song], case: TestCase) -> TestResult:
    try:
        recs = get_recommendations(songs, case.profile, k=3)
        top_song, top_score, _ = recs[0]
        top3_moods = [s.mood for s, _, _ in recs]

        genre_ok = top_song.genre == case.expected_genre
        mood_ok  = case.expected_mood in top3_moods
        passed   = genre_ok and mood_ok

        notes = []
        if not genre_ok:
            notes.append(f"expected genre '{case.expected_genre}', got '{top_song.genre}'")
        if not mood_ok:
            notes.append(f"expected mood '{case.expected_mood}' in top 3, got {top3_moods}")

        return TestResult(
            case=case, passed=passed,
            top_title=top_song.title, top_genre=top_song.genre,
            top_mood=top_song.mood, top_score=top_score,
            notes="; ".join(notes) if notes else "all checks passed",
        )
    except Exception as exc:
        return TestResult(
            case=case, passed=False,
            top_title="ERROR", top_genre="", top_mood="", top_score=0.0,
            notes=str(exc),
        )


def run_all_tests(songs: list[Song]) -> list[TestResult]:
    return [run_test(songs, case) for case in TEST_CASES]


def print_summary(results: list[TestResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    avg    = sum(r.top_score for r in results) / total if total else 0

    print("\n" + "=" * 62)
    print(f"  TEST HARNESS RESULTS — {passed}/{total} passed  |  avg top-score: {avg:.2f}")
    print("=" * 62)
    for r in results:
        icon = "✓" if r.passed else "✗"
        print(f"  {icon}  {r.case.description}")
        print(f"       top: \"{r.top_title}\" ({r.top_genre}/{r.top_mood})  score={r.top_score:.2f}")
        print(f"       {r.notes}")
    print("=" * 62)
    if passed == total:
        print("  All tests passed — scoring engine is reliable.\n")
    else:
        print(f"  {total - passed} test(s) failed — see notes above.\n")
