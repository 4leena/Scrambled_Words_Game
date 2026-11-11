"""
build_word_list.py — generate word_list.txt for speed_game.py's shared pool
===========================================================================
Run this ONCE on any machine that has a system dictionary (your Mac does:
/usr/share/dict/words). It writes word_list.txt next to speed_game.py, listing
every valid English word that can be built from the 14 shared tiles — each
letter used at most once, minimum 3 letters. Copy that word_list.txt to the
game machine alongside speed_game.py for full word coverage.

Usage:
    python3 build_word_list.py
    python3 build_word_list.py --dict /path/to/wordfile --min 3

Without word_list.txt, speed_game.py still runs on a smaller built-in set.
"""

import argparse
import os
import sys
from collections import Counter

# Keep this identical to SHARED_POOL in speed_game.py.
SHARED_POOL = list("ETAOINSHRDLUCP")

DEFAULT_DICTS = [
    "/usr/share/dict/words",            # Linux / macOS
    "/usr/share/dict/american-english",
    "/usr/dict/words",
]


def _formable(word, pool_counts, min_len):
    w = word.strip().upper()
    if len(w) < min_len or not w.isalpha():
        return False
    wc = Counter(w)
    return all(pool_counts.get(c, 0) >= n for c, n in wc.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", help="path to a word list (one word per line)")
    ap.add_argument("--min", type=int, default=3, help="minimum word length")
    ap.add_argument("--out", help="output path (default: next to this script)")
    args = ap.parse_args()

    dict_path = args.dict
    if not dict_path:
        for cand in DEFAULT_DICTS:
            if os.path.exists(cand):
                dict_path = cand
                break
    if not dict_path or not os.path.exists(dict_path):
        print("ERROR: no dictionary found. Pass one with --dict "
              "(one word per line).")
        sys.exit(1)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "word_list.txt")
    pool_counts = Counter(SHARED_POOL)

    seen = set()
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().upper()
            # Skip proper nouns / possessives / anything non-alpha.
            if w and w == line.strip().upper() and _formable(w, pool_counts, args.min):
                seen.add(w)

    words = sorted(seen)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")

    print(f"Dictionary : {dict_path}")
    print(f"Pool       : {''.join(SHARED_POOL)}")
    print(f"Min length : {args.min}")
    print(f"Wrote {len(words)} words → {out}")
    print("Copy word_list.txt next to speed_game.py on the game machine.")


if __name__ == "__main__":
    main()