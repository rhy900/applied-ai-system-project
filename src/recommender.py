from dataclasses import dataclass
import csv
import logging

logger = logging.getLogger(__name__)


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    name: str
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool = False

    def validate(self) -> list[str]:
        errors = []
        if not self.favorite_genre.strip():
            errors.append("favorite_genre cannot be empty")
        if not self.favorite_mood.strip():
            errors.append("favorite_mood cannot be empty")
        if not 0.0 <= self.target_energy <= 1.0:
            errors.append(f"target_energy must be 0–1, got {self.target_energy}")
        return errors


def load_songs_from_csv(path: str) -> list[Song]:
    songs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(Song(
                id=int(row["id"]),
                title=row["title"],
                artist=row["artist"],
                genre=row["genre"],
                mood=row["mood"],
                energy=float(row["energy"]),
                tempo_bpm=float(row["tempo_bpm"]),
                valence=float(row["valence"]),
                danceability=float(row["danceability"]),
                acousticness=float(row["acousticness"]),
            ))
    logger.info(f"Loaded {len(songs)} songs from {path}")
    return songs


def score_song(song: Song, profile: UserProfile) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []

    if song.genre == profile.favorite_genre:
        score += 2.0
        reasons.append(f"Genre match ({song.genre})")

    if song.mood == profile.favorite_mood:
        score += 1.5
        reasons.append(f"Mood match ({song.mood})")

    energy_proximity = 1.0 - abs(song.energy - profile.target_energy)
    score += energy_proximity
    reasons.append(f"Energy fit: {energy_proximity:.2f}")

    if profile.likes_acoustic and song.acousticness >= 0.6:
        score += 0.5
        reasons.append(f"Acoustic match ({song.acousticness:.2f})")

    return round(score, 4), reasons


def get_recommendations(
    songs: list[Song], profile: UserProfile, k: int = 5
) -> list[tuple[Song, float, list[str]]]:
    errors = profile.validate()
    if errors:
        raise ValueError(f"Invalid profile: {'; '.join(errors)}")

    scored = [(song, *score_song(song, profile)) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
