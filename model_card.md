# Model Card — Music Recommender Agentic AI

## Intended Use

This system is designed for educational use in AI110 to demonstrate how an agentic AI pipeline works. It is a simulation using a 20-song catalog and is not intended to replace production music services.

**Primary users:** Students, instructors, portfolio reviewers.
**Primary task:** Personalized music recommendation via natural language.

---

## System Limitations and Biases

- **Catalog size:** Only 20 songs exist in the dataset. Any genre or mood with fewer than 2–3 songs will produce limited diversity in recommendations. Folk, jazz, and r&b fans are underserved.
- **Genre dominance:** The scoring formula awards 2.0 points for a genre match — the highest single weight. A perfect-energy, perfect-mood song in the wrong genre will lose to an average song in the right genre.
- **Exact-string matching:** "chill" and "peaceful" are treated as completely different moods. There is no semantic similarity between mood labels.
- **No user history:** The system cannot learn from whether a user actually liked a recommendation. Every session starts fresh.
- **Claude's parsing bias:** When converting free text to a profile, Claude may default to mainstream genres (pop, rock) when the user's intent is ambiguous. Less common genres like synthwave require explicit mention.

---

## Potential for Misuse

- The system opens a browser tab automatically. A malicious actor could swap the YouTube URL construction for a harmful URL. The current implementation only builds `youtube.com/results?search_query=...` from song title and artist — never from user input — which prevents injection.
- The system does not store or transmit user input beyond the current session's API call.

---

## What Surprised Me During Testing

- The scoring engine was **more reliable than expected** — 6/6 test cases passed on the first run with no tuning. The genre + mood + energy proximity formula is simple but effective when the catalog covers all target genres.
- Claude **occasionally hallucinated genre names** (returned "lo-fi" instead of "lofi") during the planning step, which is why the guardrail that normalizes and validates genre output was added.
- The YouTube feature made the demo dramatically more engaging — watching a browser tab open in response to a text input made the "agent acting" concept immediately understandable.

---

## AI Collaboration Notes

**Helpful suggestion:** When I described the agentic workflow, Claude suggested separating the three steps (plan/act/check) into distinct functions rather than one large class method. This made each step individually testable and made the agent trace printed to the terminal much clearer to follow.

**Flawed suggestion:** Claude initially suggested using the `requests` library to query YouTube's Data API for video IDs, which would have required OAuth credentials and a quota-limited API key. That would have broken the demo for anyone without a Google Cloud account. The simpler `webbrowser.open()` approach with a search URL was the correct solution — no external credentials needed.

---

## Reflection

Building this project clarified the difference between a model and a system. The Claude API is just one component — the scoring engine, the guardrails, the logging, the browser integration, and the test harness are all the engineering that makes the AI output actually useful and trustworthy. A model that gives a great answer unreliably is less valuable than a simpler model embedded in a system that catches its failures. The self-evaluation step (CHECK) was the most interesting to implement: asking the AI to critique its own output surfaced limitations it would not have mentioned unprompted, which directly improved the honesty of the recommendation narrative.
