"""
generate_phrases.py
===================
Run ONCE on your Mac before camp to generate all Reachy dialogue.

Requirements:
    pip install ollama
    ollama pull gemma3:4b
    ollama serve   (in a separate terminal)

Output:
    phrases.json   — copy this to Reachy alongside reachy_feedback_server.py

Usage:
    python3 generate_phrases.py
"""

import json
import os
import time

try:
    import ollama
except ImportError:
    print("ERROR: run  pip install ollama  first.")
    raise

# ── Config ────────────────────────────────────────────────────────────────────
MODEL        = "gemma3:4b"
N_VARIATIONS = 10          # variations per key
OUT_PATH     = os.path.join(os.path.dirname(__file__), "phrases.json")

SYSTEM_PROMPT = (
    "You write short spoken lines for a cheerful robot called Reachy Mini "
    "who is coaching kids aged 6 to 10 at a summer camp word scramble game. "
    "Rules for every line you write:\n"
    "- Maximum 10 words. Shorter is better.\n"
    "- Crisp and concrete: say WHAT to do next, not vague praise.\n"
    "- NEVER use filler praise like 'fantastic effort', 'amazing job', "
    "'you're doing great', 'brilliant', or 'word detective'.\n"
    "- Never invent an answer word — the game supplies the real word itself.\n"
    "- Natural when spoken aloud — no emojis, no hashtags, no quotation marks.\n"
    "- Upbeat and playful, like a fun camp counselor. Exclamation marks OK.\n"
    "- Never start two lines the same way.\n"
    "- Output ONLY the line itself — no explanation, no numbering, no preamble."
)

# ── Phrase prompts ────────────────────────────────────────────────────────────
# Each value is a prompt describing WHAT this line should achieve.
# The model is asked for ONE line per call; we call it N_VARIATIONS times.

PROMPTS = {
    "intro": (
        "Reachy is greeting kids at the start of camp. "
        "Introduce yourself as Reachy Mini, their word coach for today."
    ),
    "intro_rules": (
        "Explain the word scramble game rules in one spoken sentence: "
        "pick a card, show the camera, unscramble the tiles, press the buzzer. "
        "Keep it clear for a 7-year-old."
    ),
    "ready_prompt": (
        "Ask the child to press the buzzer when they are ready to start."
    ),
    "session_start": (
        "The game is starting. Tell the child to grab their first card "
        "and show it to the camera."
    ),
    "waiting_for_card": (
        "Reachy is waiting for the child to pick up a word card and show it "
        "to the camera. Give a friendly nudge."
    ),
    "card_detected": (
        "Reachy just saw the word card and knows the word. "
        "Tell the child to now arrange the letter tiles and press the buzzer."
    ),
    "card_repeated": (
        "The child showed the same card they already played. "
        "Tell them to pick a different card."
    ),
    "between_words": (
        "The child just finished a word and needs to move to the next one. "
        "Give a quick encouraging bridge line."
    ),
    "observing": (
        "Reachy is watching the child arrange tiles. "
        "Say something encouraging while they work."
    ),
    "idle": (
        "The child hasn't moved the tiles in a while. "
        "Gently encourage them to keep trying."
    ),
    "idle_attempt2": (
        "The child still hasn't moved after a second nudge. "
        "Remind them they can ask for a hint."
    ),
    "wrong_letter": (
        "A letter tile is on the table that doesn't belong in the current word. "
        "Gently point out there might be a wrong letter."
    ),
    "hint": (
        "Reachy is about to give the child a hint. "
        "Say a short intro line before the hint is spoken."
    ),
    "correct": (
        "The child just unscrambled the word correctly. "
        "Give a short enthusiastic congratulation."
    ),
    "wrong_attempt1": (
        "The child buzzed but the word was wrong — first attempt. "
        "Write a TINY 2-4 word reaction like 'Not quite!' — a real hint "
        "is spoken right after your line, so keep it super short."
    ),
    "wrong_attempt2": (
        "The child got it wrong again — last attempt. "
        "Write a TINY 2-5 word final-chance line like 'One more try!' — "
        "a hint follows your line."
    ),
    "word_failed": (
        "The child used all three attempts and did not get the word. "
        "Write a kind 3-6 word consolation. Do NOT state any answer word — "
        "the game reveals the real word right after your line."
    ),
    "streak_2": (
        "The child has got 2 words correct in a row. Celebrate the streak briefly."
    ),
    "streak_3": (
        "The child has got 3 words correct in a row. React with real excitement."
    ),
    "streak_4": (
        "The child is on a massive streak of 4 or more correct words. "
        "Be genuinely amazed and excited."
    ),
    "struggling_2": (
        "The child has failed 2 words in a row. "
        "Give warm encouragement — remind them mistakes help us learn."
    ),
    "struggling_3": (
        "The child has failed 3 or more words in a row. "
        "Be extra supportive and remind them you are proud of them for not giving up."
    ),
    "level_complete": (
        "The child has finished all words in this level. "
        "Celebrate the level completion enthusiastically."
    ),
    "level_up": (
        "The child is moving to the next level with longer words. "
        "Hype them up for the challenge ahead."
    ),
    "gameover": (
        "The full game is over. "
        "Give a warm closing congratulation — they should be proud."
    ),
    "session_timeout": (
        "The 10-minute session timer has run out. "
        "Tell the child time is up and that they worked really hard."
    ),
    "restart": (
        "A new round is starting. "
        "Tell the child to grab their first card and show it to the camera."
    ),
}


# ── Generation ────────────────────────────────────────────────────────────────

def generate_variations(key: str, prompt: str, n: int) -> list[str]:
    """Call Ollama n times and collect unique variations."""
    variations = []
    seen = set()
    attempts = 0
    max_attempts = n * 3   # allow retries for duplicates

    while len(variations) < n and attempts < max_attempts:
        attempts += 1
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.9},  # high temp = more variety
            )
            line = response["message"]["content"].strip()

            # Clean up common model artefacts
            line = line.strip('"').strip("'").strip()
            if line and line not in seen:
                seen.add(line)
                variations.append(line)
                print(f"  [{len(variations)}/{n}] {line}")

        except Exception as e:
            print(f"  [WARN] attempt {attempts} failed: {e}")
            time.sleep(1)

    return variations


def main():
    print(f"Generating phrases with {MODEL} — {N_VARIATIONS} variations per key")
    print(f"Output → {OUT_PATH}\n")

    # Load existing file so we can resume if interrupted
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            output = json.load(f)
        print(f"Resuming — {len(output)} keys already done.\n")
    else:
        output = {}

    total = len(PROMPTS)
    for i, (key, prompt) in enumerate(PROMPTS.items(), 1):
        if key in output and len(output[key]) >= N_VARIATIONS:
            print(f"[{i}/{total}] SKIP {key} (already has {len(output[key])} variations)")
            continue

        print(f"\n[{i}/{total}] Generating: {key}")
        variations = generate_variations(key, prompt, N_VARIATIONS)

        if not variations:
            print(f"  [ERROR] No variations generated for {key} — skipping")
            continue

        output[key] = variations

        # Save after every key so progress is never lost
        with open(OUT_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {len(variations)} variations for '{key}'")

    print(f"\n✅ Done! {len(output)} keys written to {OUT_PATH}")
    print(f"Next step: scp phrases.json pollen@10.42.0.72:~/Scramble_Words/")


if __name__ == "__main__":
    main()