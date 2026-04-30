"""
Agentic workflow:
  STEP 1 — PLAN:     Gemini reads the user's natural-language request, reasons
                     about their context, and decides genre/mood/energy profile.
  STEP 2 — RETRIEVE: TF-IDF RAG retrieves the 3 most relevant song descriptions
                     from catalog text documents to ground the CHECK response.
  STEP 3 — ACT:      Scoring engine retrieves top songs. Agent opens the best
                     match's YouTube search in the browser.
  STEP 4 — CHECK:    Gemini reviews its own recommendation using the retrieved
                     song descriptions and explains the choice with context.
"""

import os
import webbrowser
import logging
from dataclasses import dataclass, field
from urllib.parse import quote_plus

from google import genai

from src.recommender import Song, UserProfile, get_recommendations
from src.retriever import SongDoc, load_descriptions, retrieve as rag_retrieve
from src.logger import log_guardrail

logger = logging.getLogger(__name__)

VALID_GENRES = {"pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop", "r&b", "folk", "electronic"}
VALID_MOODS  = {"happy", "chill", "intense", "moody", "relaxed", "focused", "melancholy", "peaceful", "energetic"}
MODEL = "gemini-2.0-flash"

_DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "song_descriptions.txt")


@dataclass
class AgentResult:
    profile: UserProfile | None
    top_song: Song | None = None
    top_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    runner_up: Song | None = None
    youtube_url: str = ""
    plan_text: str = ""
    reasoning: str = ""
    check_text: str = ""
    opened_browser: bool = False
    retrieved_docs: list[SongDoc] = field(default_factory=list)


def _youtube_url(song: Song) -> str:
    query = quote_plus(f"{song.title} {song.artist}")
    return f"https://www.youtube.com/results?search_query={query}"


def _ask(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


# ── STEP 1: PLAN ─────────────────────────────────────────────────────────────

def plan(client: genai.Client, user_text: str) -> tuple[UserProfile, str, str]:
    """
    Ask Gemini to reason about the request, then extract a structured UserProfile.
    Returns (profile, plan_note, reasoning).
    """
    prompt = f"""You are a music taste expert. Analyze the user's request step by step.

Output ONLY this block — no extra text:

REASON: <1-2 sentences analyzing the user's emotional state and listening context>
NAME: <first name or "Listener">
GENRE: <exactly one of: pop | lofi | rock | ambient | jazz | synthwave | indie pop | r&b | folk | electronic>
MOOD: <exactly one of: happy | chill | intense | moody | relaxed | focused | melancholy | peaceful | energetic>
ENERGY: <float 0.0–1.0 — low energy = 0.1–0.3, medium = 0.4–0.6, high = 0.7–1.0>
ACOUSTIC: <true | false>
PLAN_NOTE: <one sentence explaining your genre/mood/energy choice>

User request: "{user_text}" """

    text = _ask(client, prompt)
    logger.info(f"PLAN response:\n{text}")

    lines: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            lines[key.strip().upper()] = val.strip()

    genre = lines.get("GENRE", "pop").lower()
    mood  = lines.get("MOOD", "happy").lower()

    if genre not in VALID_GENRES:
        log_guardrail(logger, f"Invalid genre '{genre}' from Gemini — defaulting to 'pop'")
        genre = "pop"
    if mood not in VALID_MOODS:
        log_guardrail(logger, f"Invalid mood '{mood}' from Gemini — defaulting to 'happy'")
        mood = "happy"

    try:
        energy = float(lines.get("ENERGY", "0.5"))
        energy = max(0.0, min(1.0, energy))
    except ValueError:
        log_guardrail(logger, "Could not parse ENERGY — defaulting to 0.5")
        energy = 0.5

    profile = UserProfile(
        name=lines.get("NAME", "Listener"),
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=lines.get("ACOUSTIC", "false").lower() == "true",
    )
    reasoning  = lines.get("REASON", "")
    plan_note  = lines.get("PLAN_NOTE", "")
    return profile, plan_note, reasoning


# ── STEP 2: RETRIEVE (RAG) ────────────────────────────────────────────────────

def retrieve_context(song_docs: list[SongDoc], query: str, k: int = 3) -> list[SongDoc]:
    """Return the top-k catalog documents most relevant to the user's query."""
    return rag_retrieve(song_docs, query, k=k)


# ── STEP 3: ACT ──────────────────────────────────────────────────────────────

def act(songs: list[Song], profile: UserProfile, open_browser: bool = True) -> tuple[Song, float, list[str], Song | None, str]:
    """Retrieve top songs and open the best match on YouTube."""
    recs = get_recommendations(songs, profile, k=5)
    top_song, top_score, reasons = recs[0]
    runner_up = recs[1][0] if len(recs) > 1 else None

    url = _youtube_url(top_song)
    logger.info(f"ACT — top pick: '{top_song.title}' by {top_song.artist} (score={top_score})")

    if open_browser:
        webbrowser.open(url)
        logger.info("ACT — browser opened")

    return top_song, top_score, reasons, runner_up, url


# ── STEP 4: CHECK ─────────────────────────────────────────────────────────────

def check(
    client: genai.Client,
    profile: UserProfile,
    top_song: Song,
    reasons: list[str],
    runner_up: Song | None,
    retrieved_docs: list[SongDoc] | None = None,
) -> str:
    """Gemini reviews its own recommendation, grounded by retrieved descriptions."""
    runner_line = (
        f'Runner-up: "{runner_up.title}" by {runner_up.artist} ({runner_up.genre}/{runner_up.mood})'
        if runner_up else "Runner-up: none"
    )

    context_block = ""
    if retrieved_docs:
        entries = "\n".join(
            f'  • "{d.title}" by {d.artist}: {d.description[:150]}…'
            for d in retrieved_docs
        )
        context_block = f"\nRetrieved catalog context (use this to enrich your explanation):\n{entries}\n"

    prompt = f"""You recommended music to {profile.name}.

Their vibe: {profile.favorite_genre} / {profile.favorite_mood} / energy {profile.target_energy}

Your top pick: "{top_song.title}" by {top_song.artist}
  Genre: {top_song.genre} | Mood: {top_song.mood} | Energy: {top_song.energy}
  Match reasons: {', '.join(reasons)}
{runner_line}
{context_block}
In 2–3 friendly sentences: explain why this is a great pick for them, mention the runner-up as an alternative, and honestly note any limitation (e.g. small catalog, no exact mood match). Use the retrieved context to make your explanation specific and vivid."""

    return _ask(client, prompt)


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

class MusicAgent:
    def __init__(self, songs: list[Song], api_key: str | None = None):
        self.songs = songs
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        try:
            self.song_docs = load_descriptions(_DESCRIPTIONS_PATH)
        except FileNotFoundError:
            logger.warning("song_descriptions.txt not found — RAG step will be skipped")
            self.song_docs = []

    def run(self, user_input: str, open_browser: bool = True) -> AgentResult:
        result = AgentResult(profile=None)

        # STEP 1 — PLAN
        print("\n[STEP 1 — PLAN] Analyzing your request...")
        profile, plan_note, reasoning = plan(self.client, user_input)
        result.profile   = profile
        result.plan_text = plan_note
        result.reasoning = reasoning
        print(f"  Reasoning: {reasoning}")
        print(f"  Taste profile → genre: {profile.favorite_genre} | mood: {profile.favorite_mood} | energy: {profile.target_energy}")
        print(f"  Agent note: {plan_note}")

        # STEP 2 — RETRIEVE (RAG)
        print("\n[STEP 2 — RETRIEVE] Fetching relevant song context from catalog documents...")
        retrieved = retrieve_context(self.song_docs, user_input, k=3) if self.song_docs else []
        result.retrieved_docs = retrieved
        if retrieved:
            for doc in retrieved:
                print(f"  Retrieved: \"{doc.title}\" by {doc.artist}")
        else:
            print("  (No descriptions file — RAG skipped)")

        # STEP 3 — ACT
        print("\n[STEP 3 — ACT] Finding your song and opening YouTube...")
        top_song, top_score, reasons, runner_up, url = act(self.songs, profile, open_browser=open_browser)
        result.top_song       = top_song
        result.top_score      = top_score
        result.reasons        = reasons
        result.runner_up      = runner_up
        result.youtube_url    = url
        result.opened_browser = open_browser
        print(f"  Top pick → \"{top_song.title}\" by {top_song.artist}  (score: {top_score})")
        if runner_up:
            print(f"  Runner-up → \"{runner_up.title}\" by {runner_up.artist}")
        print(f"  YouTube: {url}")

        # STEP 4 — CHECK
        print("\n[STEP 4 — CHECK] Evaluating recommendation quality with retrieved context...")
        check_text = check(self.client, profile, top_song, reasons, runner_up, retrieved)
        result.check_text = check_text
        print(f"  {check_text}")

        return result
