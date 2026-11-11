import json
import random
import socket
import threading
import time
import atexit
import platform
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np

try:
    from session_recorder import SessionRecorder
except Exception as _e:
    SessionRecorder = None
    print(f"[REC] session_recorder unavailable ({_e}) — running without recording.")

try:
    import tkinter as tk
    _TK_OK = True
except ImportError:
    _TK_OK = False
    print("tkinter not available — cannot run GUI.")

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

from detection import (detect_markers, load_templates, create_marker_id_dict,
                       detect_player_ids, draw_player_markers,
                       detect_play_area, filter_markers_by_play_area,
                       draw_play_area, get_all_aruco_ids, hull_from_points,
                       detect_word_card, draw_word_card_overlay,
                       build_word_card_map,
                       detect_card_area, draw_card_area)
from game_logger import SessionLogger

# ── Player names (player_names.json) ─────────────────────────────────────────
try:
    from player_names import get_player_name
except ImportError:
    def get_player_name(player_id):
        return "Player" if player_id is None else f"Player {player_id}"
    print("player_names.py not found — using generic player labels.")

# ── Buzzers (GPIO buttons) ────────────────────────────────────────────────────
# GREEN buzzer = SPACE (start / submit word). RED buzzer = end the session
# gracefully (score + goodbye + kiosk reset) — NOT a hard app quit, so a kid
# pressing it can't kill the unattended kiosk for the day.
BUZZER_PIN      = 20   # green — acts as SPACE
QUIT_BUZZER_PIN = 21   # red   — ends the current game session

_buzzer_event = threading.Event()
_quit_event   = threading.Event()

try:
    from gpiozero import Button as _GPIOButton
    _buzzer_btn = _GPIOButton(BUZZER_PIN, pull_up=False, bounce_time=0.05)
    _buzzer_btn.when_pressed = _buzzer_event.set
    print(f"Green buzzer (SPACE) ready on GPIO {BUZZER_PIN}.")
    try:
        _quit_btn = _GPIOButton(QUIT_BUZZER_PIN, pull_up=False, bounce_time=0.05)
        _quit_btn.when_pressed = _quit_event.set
        print(f"Red buzzer (end game) ready on GPIO {QUIT_BUZZER_PIN}.")
    except Exception as _e2:
        _quit_btn = None
        print(f"No red buzzer available ({_e2}) — green/keyboard only.")
except Exception as _e:
    _buzzer_btn = None
    _quit_btn   = None
    print(f"No GPIO buzzer available ({_e}) — keyboard-only mode.")

# -----------------------------
# CONFIG
# -----------------------------

# Players face the camera from the other side of the letter cards, so the word
# must be read right-to-left from the camera's point of view.
REVERSE_READING = True

REACHY_FEEDBACK_URL = "http://reachy-mini.local:8765/event"
REACHY_ENABLED = True

# How often (seconds) this PC mirrors the robot's spoken-feedback transcript to
# its own logs/ folder, so an unattended "study in the wild" collects the
# transcript automatically with no manual scp. The robot keeps its own copy
# too, so nothing is lost if this PC is off.
SPEECH_SYNC_S = 45

# Recording-camera capture size/framerate. With software encoding (libx264 on
# the CPU), the encoder MUST keep up with real time or the recording falls
# progressively behind — a delay that grows the longer it records. A modest
# size + framerate keeps encoding light enough to stay real-time on this PC
# while it also runs ArUco detection and the game. Raise these only if the
# machine comfortably keeps up (watch for the delay returning).
RECORD_CAM_SIZE = "640x480"
RECORD_CAM_FPS  = 20

# ── Dual-camera login detection ───────────────────────────────────────────────
# During the scan screen, player ID cards are detected on BOTH the external
# game camera and the Reachy Mini's own camera (served at GET /player_ids by
# reachy_feedback_server.py).
ROBOT_ID_POLL_S = 0.30   # how often we poll the robot camera for IDs
ROBOT_ID_TTL_S  = 1.0    # robot-seen ID counts as "present" for this long

# Auto-start: player ID cards are detected on the scan screen; login is
# sticky (one detection = logged in — the card can go down immediately).
# The game WAITS for BOTH players: after player 1 logs in there is no
# countdown, just a "Player 2 — show your card!" prompt; the start countdown
# only arms once the second card is seen. SPACE / buzzer / the Skip button
# remain the manual escape hatch to play solo.
AUTO_START_TWO_PLAYERS_S = 1.5   # both players in → start after this

# ── Pre-login "seated and ready?" screen ─────────────────────────────────────
# Mirrors speed_game.py: before we scan any ID cards, the kiosk shows a ready
# screen and waits for a buzzer press. This gives kids a moment to sit down
# and settle before the robot starts asking for cards, instead of the scan
# screen shouting at them the instant they walk past.
#
# The spoken invite is FACE-GATED, unlike speed_game.py, which fires its
# ready_prompt unconditionally the moment the screen is built. game.py runs
# unattended with a kiosk auto-reset, so an unconditional prompt would have
# the robot announcing "press the buzzer when you're seated" to an empty room
# on every reset cycle.
#
# The gating lives on the ROBOT, not here: its face_greet logic already does
# exactly this job (standby wake, 30s cooldown, busy check) for the scan
# screen. We just declare which screen is up via seated_prompt / scan_id_prompt
# and let face_greet pick the matching line. Doing it here as well produced
# TWO invites racing: face_greet's "show me your name card" and ours.

# ── Inactivity watchdog ───────────────────────────────────────────────────────
# If the ROBOT's camera sees no face AND the game camera sees no activity
# (no tiles, no cards, no buzzer) for this long during play, the session ends
# automatically with a goodbye.
INACTIVITY_TIMEOUT_S = 90

# ── Kiosk mode: unattended in-the-wild deployment ────────────────────────────
# After ANY game ending (finished, session timer, inactivity), the game
# automatically returns to the scan/login screen after this many seconds, so
# the next camper can walk up, show their ID card, and play — no staff needed
# to stop or restart anything. Pressing New Game or starting a session
# manually cancels a pending reset.
KIOSK_RESET_DELAY_S = 120

WORDS_PER_ROUND      = 10   # fallback — overridden per category below
MAX_ATTEMPTS_PER_WORD = 3
SESSION_SECONDS      = 8 * 60   # 8-minute session
DEFAULT_CATEGORY     = "animals"
STABLE_FRAMES        = 5   # frames a reading must persist before invalid-letter fires

# How long to ignore the card area after finishing a word. The card just
# played is still lying there while the robot says "bring a new card", and
# without this the game reads it as a fresh scan and scolds the kid for
# repeating it. See advance_to_next_word() for the second, more important
# guard that goes with this one.
CARD_GRACE_S         = 2.0

# ── Hands-away buzzer grace ──────────────────────────────────────────────────
# Ported from speed_game.py, which has had this and game.py never did — the
# cause of correct words being judged wrong on the first attempt.
#
# Kids press the buzzer with a hand/arm still over the tiles, so the camera
# reads an OCCLUDED word at the instant of the press: lay out C-A-T, press,
# and the arm covers the T, so _check() judges "CA" and calls it wrong.
#
# Stability alone is NOT sufficient: a RESTING arm produces a partial reading
# that is perfectly steady frame-to-frame, so it sails through a stability
# check while still covering tiles. Fix: after every press, wait a fixed
# HANDS_DELAY_S first (hands naturally retract right after pressing), and only
# THEN require a stable reading. The word is judged from that settled view.
HANDS_DELAY_S       = 2.0   # pause after the press before reading (slow path)
HANDS_STABLE_FRAMES = 3     # consecutive identical frames = clean view
HANDS_WAIT_MAX_S    = 3.0   # extra time after the delay to find a stable view
HANDS_POLL_MS       = 80    # how often the hands-wait poller re-checks
# Fast path: if the reading was ALREADY this stable at the moment of the press,
# the tiles were clearly untouched and we judge immediately with no delay. The
# bar is deliberately much higher than HANDS_STABLE_FRAMES — a resting arm can
# hold a partial reading still for 3 frames, but not for ~8 (~0.5s+). Raise
# this if you still see arms judged as words; lower it for snappier feedback.
HANDS_FAST_STABLE_FRAMES = 8

# Frames to grab-and-discard before each real read, to clear anything the
# driver buffered while we were busy detecting — the ~30 fps camera outruns
# the ~100 ms detection pass, so without this cap.read() returns stale frames
# and the view lag GROWS the longer a turn runs. grab() only decodes the
# header, so draining is cheap.
CAM_DRAIN_FRAMES = 2

# ── Pre-buzzer nudge pacing ──────────────────────────────────────────────────
# These nudges fire while the kid is still working, BEFORE any buzzer press,
# so they have to stay rare enough not to talk over the real verdict.
#
# BUZZER_HINT_DELAY_S is the important one. The reminder used to fire the
# moment the tiles spelled the target — which is the moment the kid reaches
# for the buzzer — so it was still playing when the press landed and the
# "Woo hoo!" stacked on top of it. Holding it for a few seconds first means a
# kid who presses promptly never hears it, and the only kid who does hear it
# is one who has stalled and isn't about to press anyway.
#
# The SCREEN is not gated by any of this: "Looks right! Press the buzzer!"
# turns green the instant the tiles are right. Instant visual, patient voice.
BUZZER_HINT_DELAY_S   = 5.0    # tiles correct this long before we say anything
BUZZER_HINT_REPEAT_S  = 20.0   # then at most this often (was 8s)
WRONG_LETTER_REPEAT_S = 25.0   # spoken "check your tiles" gap (was every change)

# Words per round — animals only now, use every word in the list
WORDS_PER_LEVEL = {
    "animals": 29,   # all 12 three-letter + 17 four-letter printed cards
}

# -----------------------------
# LETTER SET CONSTRAINT
# The physical tile set covers A through Y (one tile per letter, no Z).
# Every word in the game must:
#   1. Use only these letters.
#   2. Never repeat the same letter (one physical tile per letter).
# -----------------------------
ALLOWED_LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXY')


def _tile_valid(word):
    """Return True if word uses only ALLOWED_LETTERS with no repeated letter."""
    w = word.upper()
    return all(c in ALLOWED_LETTERS for c in w) and len(set(w)) == len(w)


# -----------------------------
# WORD BANK  (single source of truth)
# ANIMAL_CARDS mirrors the PRINTED card deck EXACTLY: index in this list ==
# ArUco card ID (animals offset is 0 in detection.build_word_card_map), and
# the second element is the scramble PRINTED on that physical card — the
# screen must show the same letters the kid is holding, so we never
# re-scramble randomly. If the deck is ever reprinted, update this list to
# match the new PDF.
# -----------------------------
ANIMAL_CARDS = [
    # (word, printed_scramble)          # card ID
    ("ant",  "ATN"),    # 0
    ("bat",  "ATB"),    # 1
    ("cat",  "ATC"),    # 2
    ("cod",  "ODC"),    # 3
    ("cow",  "WOC"),    # 4
    ("doe",  "ODE"),    # 5
    ("dog",  "ODG"),    # 6
    ("hen",  "NHE"),    # 7
    ("hog",  "GOH"),    # 8
    ("owl",  "OLW"),    # 9
    ("ram",  "ARM"),    # 10
    ("rat",  "RTA"),    # 11
    ("bear", "BAER"),   # 12
    ("boar", "RBOA"),   # 13
    ("clam", "LMAC"),   # 14
    ("colt", "TLCO"),   # 15
    ("crab", "CBRA"),   # 16
    ("crow", "CORW"),   # 17
    ("gnat", "ATGN"),   # 18
    ("goat", "OTAG"),   # 19
    ("hare", "RHEA"),   # 20
    ("lamb", "LMBA"),   # 21
    ("mare", "ERMA"),   # 22
    ("mole", "OEML"),   # 23
    ("moth", "OMTH"),   # 24
    ("newt", "EWNT"),   # 25
    ("toad", "DTAO"),   # 26
    ("worm", "OMWR"),   # 27
    ("wren", "RNEW"),   # 28
]

ANIMAL_WORDS = [w for w, _ in ANIMAL_CARDS]

# printed scramble lookup: word -> exactly what's on the physical card
PRINTED_SCRAMBLES = {w: s for w, s in ANIMAL_CARDS}

_invalid = [w for w in ANIMAL_WORDS if not _tile_valid(w)]
assert not _invalid, f"Words violate the tile constraint: {_invalid}"
_bad_scrambles = [w for w, s in ANIMAL_CARDS if sorted(w.upper()) != sorted(s.upper())]
assert not _bad_scrambles, f"Printed scramble doesn't match its word: {_bad_scrambles}"

WORD_BANK = ANIMAL_WORDS              # dictionary used for anagram matching
WORDS     = frozenset(WORD_BANK)

# -----------------------------
# CATEGORIES  (single level — animals, mix of 3 and 4 letter words)
# -----------------------------
CATEGORIES = {
    "animals": ANIMAL_WORDS,
}



WORD_CLUES = {
    # Kid-friendly clues for every printed card word. These MUST stay in
    # sync with _ANIMAL_CLUES in reachy_feedback_server.py — the robot's
    # audio for these lines is pre-generated from that copy.
    "ant":  "A tiny insect that lives in colonies and can carry things much bigger than itself.",
    "bat":  "A flying mammal that comes out at night.",
    "cat":  "A furry pet with sharp claws and whiskers that purrs when it is happy.",
    "cod":  "A large fish that lives in cold ocean water. Fish and chips are often made from cod!",
    "cow":  "A farm animal that gives us milk.",
    "doe":  "A female deer with soft brown fur. Does are very gentle and take great care of their fawns.",
    "dog":  "A loyal pet that barks and wags its tail.",
    "hen":  "A female chicken that clucks and lays the eggs we eat for breakfast.",
    "hog":  "Another word for a pig.",
    "owl":  "A bird that hoots and is awake at night.",
    "ram":  "A male sheep with big curly horns.",
    "rat":  "A clever rodent with a long tail. Rats are actually very smart and can be trained to do tricks!",
    "bear": "A big, furry animal that loves honey and sleeps all winter.",
    "boar": "A wild pig with sharp tusks.",
    "clam": "A sea creature that lives inside two shells.",
    "colt": "A young male horse that is still growing up.",
    "crab": "A sea creature that walks sideways and has pincers.",
    "crow": "A clever black bird.",
    "gnat": "A tiny flying bug.",
    "goat": "A farm animal that loves to climb and eat almost anything.",
    "hare": "A fast-running animal like a big rabbit, with long ears and powerful back legs.",
    "lamb": "A baby sheep.",
    "mare": "A female horse.",
    "mole": "A small animal that digs tunnels underground.",
    "moth": "A flying insect like a butterfly that comes out at night.",
    "newt": "A small amphibian related to salamanders.",
    "toad": "A bumpy-skinned cousin of the frog that hops around gardens and ponds.",
    "worm": "A small, wiggly creature that lives in the soil.",
    "wren": "A tiny bird with a surprisingly loud song.",
}


# -----------------------------
# LEVELS  (single round — animals only, mix of 3 and 4 letter words)
# -----------------------------
LEVELS = [
    {
        'name':     'Animals',
        'category': 'animals',
        'color':    '#4ade80',
        'desc':     'Animals — 3 and 4 letter words',
    },
]

# Colour palette (hex, for tkinter)
P = {
    'bg':      '#0d0d0d',
    'panel':   '#141414',
    'card':    '#1c1c1c',
    'text':    '#d4d4d4',
    'dim':     '#484848',
    'sep':     '#242424',
    'green':   '#4ade80',
    'gold':    '#fbbf24',
    'red':     '#f87171',
    'blue':    '#60a5fa',
    'purple':  '#a78bfa',
    'bg_cur':  '#0b1f0e',
    'bg_ok':   '#091509',
    'bg_fail': '#1a0b0b',
}

# -----------------------------
# REACHY FEEDBACK CLIENT
# -----------------------------
class ReachyFeedbackClient:
    """Fire-and-forget event sender. Never blocks or crashes the game."""

    def __init__(self, url=REACHY_FEEDBACK_URL, enabled=REACHY_ENABLED):
        self.enabled      = enabled
        self.hostname_url = url
        self.url          = url
        self.last_sent    = {}
        self.failures     = 0
        # Reachability cache: avoid re-probing the network on every single event.
        self._reachable       = None      # None = unknown yet
        self._reachable_until = 0.0       # re-check after this time
        self._resolve()

    def _resolve(self):
        self.url = self.hostname_url
        self.failures = 0
        print(f"Reachy using: {self.url}")

    def _host_reachable(self, timeout=0.3):
        """Fast TCP check to host:port. Returns quickly instead of letting
        urlopen hang on a slow '.local' (mDNS) lookup — the urlopen `timeout`
        does NOT cover DNS resolution, so a missing robot can otherwise stall
        for a long time. Result is cached for a few seconds to avoid probing
        on every event.
        """
        now = time.time()
        if self._reachable is not None and now < self._reachable_until:
            return self._reachable
        ok = False
        try:
            p = urlparse(self.url)
            host = p.hostname
            port = p.port or (443 if p.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=timeout):
                ok = True
        except Exception:
            ok = False
        self._reachable = ok
        # If reachable, re-check occasionally (robot could drop off). If not,
        # re-check a bit more often so it reconnects promptly once plugged in.
        self._reachable_until = now + (10.0 if ok else 3.0)
        return ok

    def _ids_url(self):
        """URL of the robot's player-ID/presence endpoint (…/player_ids)."""
        p = urlparse(self.url)
        return f"{p.scheme}://{p.netloc}/player_ids"

    def _speech_log_url(self):
        """URL of the robot's spoken-feedback transcript (…/speech_log)."""
        p = urlparse(self.url)
        return f"{p.scheme}://{p.netloc}/speech_log"

    def fetch_speech_log(self):
        """Return the robot's current spoken-feedback transcript as text, or
        None if the robot isn't reachable. Short timeout; call from a
        background thread, never the UI thread."""
        if not self.enabled or not self._host_reachable():
            return None
        try:
            with urllib.request.urlopen(self._speech_log_url(), timeout=1.5) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            return None

    def get_presence(self):
        """Fetch presence info from the ROBOT's camera.

        Returns {"ids": [player ids], "face_seen_ago": secs-or-None}.
        Blocking (short timeout) — call from a background thread, never the
        GUI thread. Degrades to empty/None on any failure so the game falls
        back to external-camera-only detection.
        """
        if not self.enabled:
            return {"ids": [], "face_seen_ago": None}
        if not self._host_reachable():
            return {"ids": [], "face_seen_ago": None}
        try:
            with urllib.request.urlopen(self._ids_url(), timeout=0.5) as r:
                data = json.loads(r.read().decode())
            return {
                "ids":           [int(i) for i in data.get("ids", [])],
                "face_seen_ago": data.get("face_seen_ago"),
            }
        except Exception:
            return {"ids": [], "face_seen_ago": None}

    # Per-event cooldowns — prevent spam from camera loop firing every frame
    _COOLDOWNS = {
        "gaze":            0.5,
        "observing":       12.0,
        "waiting_for_card": 8.0,
        "wrong_letter":    5.0,
        "idle":            20.0,
        "between_words":   8.0,
        "card_detected":   2.0,   # once per card show
        "card_repeated":   4.0,
        "scan_id_prompt":  6.0,   # login prompt — fired once per scan screen
        "player_detected": 0.0,   # 0 = never throttled; already deduped per-player
        "default":         0.5,
    }

    def send(self, event_type, **payload):
        if not self.enabled:
            return
        now      = time.time()
        cooldown = self._COOLDOWNS.get(event_type, self._COOLDOWNS["default"])
        if now - self.last_sent.get(event_type, 0) < cooldown:
            return
        self.last_sent[event_type] = now

        data = json.dumps({"type": event_type, **payload}).encode("utf-8")

        def _post():
            # Skip fast if the robot isn't reachable — avoids slow '.local'
            # lookups piling up in the background and spamming the log.
            if not self._host_reachable():
                return
            try:
                req = urllib.request.Request(
                    self.url, data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=2.0).read()
                self.failures = 0
                print(f"[REACHY OK] {event_type}")
            except Exception as e:
                print(f"[REACHY FAIL] {event_type}: {e}")
                self.failures += 1
                if self.failures >= 10:
                    self.failures = 0
                    self._resolve()

        threading.Thread(target=_post, daemon=True).start()


# -----------------------------
# WORD HELPERS
# -----------------------------
def scramble(word):
    letters = list(word.upper())
    for _ in range(100):
        random.shuffle(letters)
        if "".join(letters).lower() != word.lower():
            break
    return "".join(letters)


def get_valid_anagrams(word):
    key = sorted(word.lower())
    anagrams = {w for w in WORDS if sorted(w) == key}
    anagrams.add(word.lower())
    return anagrams


_LEVEL_LENGTH_RANGE = {
    "animals": (3, 4),   # mix of 3-letter and 4-letter animal words
}


def pick_words(category):
    n   = WORDS_PER_LEVEL.get(category, WORDS_PER_ROUND)
    lo, hi = _LEVEL_LENGTH_RANGE.get(category, (3, 6))
    candidates = [w for w in CATEGORIES.get(category, []) if lo <= len(w) <= hi]
    if len(candidates) < n:
        candidates = [w for w in WORD_BANK if lo <= len(w) <= hi]
    picked = random.sample(candidates, min(n, len(candidates)))
    print(f"Category: {category}  length={lo}  n={n}  words: {picked}")
    return picked


def build_hint(target, level, detected=""):
    t = target.upper()
    d = (detected or "").upper()

    if level <= 1:
        clue = WORD_CLUES.get(target.lower())
        if clue:
            return f"Here is a clue. {clue}"
        if d and d[0] == t[0]:
            return f"Good start! The word ends with {t[-1]}."
        return f"The word starts with {t[0]}."

    if level == 2:
        correct = sum(1 for a, b in zip(d, t) if a == b) if len(d) == len(t) else 0
        praise = f"You already have {correct} letters in the right spot. " if correct else ""
        return f"{praise}It starts with {t[0]} and ends with {t[-1]}."

    return f"The word was {t}."


def precache_lines(words):
    lines = []
    for w in words:
        t = w.upper()
        clue = WORD_CLUES.get(w.lower())
        cores = [f"The word starts with {t[0]}.",
                 f"Good start! The word ends with {t[-1]}.",
                 f"It starts with {t[0]} and ends with {t[-1]}.",
                 f"The word was {t}."]
        for count in (1, 2, 3):
            cores.append(f"You already have {count} letters in the right spot. "
                         f"It starts with {t[0]} and ends with {t[-1]}.")
        if clue:
            cores.append(f"Here is a clue. {clue}")
        lines.extend(cores)
        lines.extend(f"Here is a hint. {c}" for c in cores
                     if not c.startswith("Here is a clue"))
        lines.append(f"That one was tricky. The word was {t}. "
                     "Clear the letters, and show me the next puzzle.")
    return lines


# -----------------------------
# GAME STATE
# -----------------------------
def new_state(category):
    n = WORDS_PER_LEVEL.get(category, WORDS_PER_ROUND)
    return {
        "words":                 [],     # filled one at a time as cards are shown
        "round_letters":         set(),
        "category":              category,
        "words_per_round":       n,
        "current_idx":           0,
        "done":                  [],
        "attempts":              [],
        "score":                 0,
        "detected_word":         "",
        "last_detected":         "",
        "feedback":              None,
        "feedback_timer":        0,
        "invalid_feedback":      None,
        "invalid_feedback_timer": 0,
        "tailored_feedback":     None,
        "tailored_timer":        0,
        "seen_letters_for_word": [],
        "await_clear":           False,
        "waiting_for_ready":     True,
        "waiting_for_card":      False,  # True when waiting for kid to show card
        "session_start_time":    None,
        "session_done":          False,
        "player_ids":            [],
        # ── performance tracking ──────────────────────────────────────────────
        "streak":                0,
        "wrong_streak":          0,
        "word_start_time":       None,
        "card_word":             None,   # word detected from ArUco card
        "used_words":            set(),  # words already played this round
    }


def add_card_word(state, word):
    """Add a word from a detected card as the current puzzle."""
    word = word.lower()
    if word in state["used_words"]:
        return False
    if len(state["words"]) >= state["words_per_round"]:
        return False
    state["used_words"].add(word)
    # Use the scramble PRINTED on the physical card so the screen matches
    # the letters in the kid's hand; random scramble only as a fallback.
    state["words"].append((word, PRINTED_SCRAMBLES.get(word, scramble(word))))
    state["done"].append(False)
    state["attempts"].append(0)
    state["seen_letters_for_word"].append(False)
    state["round_letters"]    = set("".join(w for w, _ in state["words"]).upper())
    state["waiting_for_card"] = False
    state["word_start_time"]  = None
    return True


def get_time_remaining(state):
    if state.get("session_done"):
        return 0
    start = state.get("session_start_time")
    if not start:
        return SESSION_SECONDS
    return max(0, SESSION_SECONDS - int(time.time() - start))


def check_invalid_letter(state):
    idx = state["current_idx"]
    detected = state["detected_word"].upper()
    # Bug fix: use len(words) not words_per_round (words list is now dynamic)
    if idx >= len(state["words"]) or not detected:
        return False
    allowed = set(state["words"][idx][1].upper())
    return any(ch not in allowed for ch in detected)


def advance_to_next_word(state):
    # The word we just finished. Its card is still physically sitting in the
    # card area — the kid hasn't picked it up yet — so we have to remember it
    # and NOT treat seeing it as "you already played that one".
    _idx_done = state["current_idx"]
    _just_played = (state["words"][_idx_done][0].lower()
                    if _idx_done < len(state["words"]) else None)

    state["current_idx"] += 1
    words_done = state["current_idx"]
    if words_done >= state["words_per_round"]:
        state["feedback"] = "gameover"
        state["feedback_timer"] = time.time()
    else:
        state["detected_word"]     = ""
        state["last_detected"]     = ""
        state["feedback"]          = None
        state["tailored_feedback"] = None
        state["tailored_timer"]    = 0
        state["await_clear"]       = True
        state["word_start_time"]   = None
        state["waiting_for_card"]  = True   # wait for next card
        state["card_word"]         = None
        state["_idle_nudge_reset"] = True

        # ── Don't scold the kid for the card they just finished ────────────
        # Setting card_word = None above makes the still-present card look
        # BRAND NEW to the camera loop (card_word != prev), so ~8 frames
        # (~0.3s) later _on_card_detected fires with the word we just played,
        # finds it in used_words, and the robot says "Oops! We did CAT
        # already!" — right on top of "Bring a new card to the playing area!".
        #
        # Two guards, because the obvious one isn't enough on its own:
        #   1. card_grace_until — ignore the card area entirely for a couple
        #      of seconds while the robot is still talking.
        #   2. awaiting_card_removal — a grace window alone only helps if the
        #      kid clears the card within it. They won't: they're listening to
        #      the robot, then reaching. So we ALSO suppress the warning for
        #      this specific word until the card area has been seen EMPTY at
        #      least once. Its card lying there is the expected state, not a
        #      mistake. Once they lift it, bringing it BACK is a real repeat
        #      and gets the real warning.
        state["card_grace_until"]      = time.time() + CARD_GRACE_S
        state["awaiting_card_removal"] = _just_played


def check_word(state):
    idx = state["current_idx"]
    if state.get("waiting_for_card"):
        return {"result": "none"}   # no card shown yet
    if idx >= len(state["words"]) or state.get("session_done"):
        return {"result": "none"}

    target   = state["words"][idx][0]
    detected = state["detected_word"].lower()
    anagrams = get_valid_anagrams(target)

    if detected in anagrams or detected[::-1] in anagrams:
        points = 3 if len(target) == 3 else 5   # 3-letter = 3 pts, 4-letter = 5 pts
        state["done"][idx]          = True
        state["score"]             += points
        state["feedback"]           = "correct"
        state["feedback_timer"]     = time.time()
        state["invalid_feedback"]   = None
        state["tailored_feedback"]  = None
        # streak tracking
        state["streak"]        = state.get("streak", 0) + 1
        state["wrong_streak"]  = 0
        state["word_start_time"] = None
        streak = state["streak"]
        advance_to_next_word(state)
        return {"result": "correct", "target": target,
                "streak": streak,
                "points": points,
                "score": state["score"],
                "total": state["words_per_round"],
                "finished": state["current_idx"] >= state["words_per_round"]}

    state["attempts"][idx] += 1
    attempt = state["attempts"][idx]
    state["feedback"]         = "wrong"
    state["feedback_timer"]   = time.time()
    state["invalid_feedback"] = None

    if attempt >= MAX_ATTEMPTS_PER_WORD:
        state["streak"]       = 0
        state["wrong_streak"] = state.get("wrong_streak", 0) + 1
        state["word_start_time"] = None
        wrong_streak = state["wrong_streak"]
        state["tailored_feedback"] = build_hint(target, 3)
        state["tailored_timer"]    = time.time()
        advance_to_next_word(state)
        return {"result": "failed", "target": target,
                "wrong_streak": wrong_streak,
                "say": (f"That one was tricky. The word was {target.upper()}. "
                        "Clear the letters, and show me the next puzzle."),
                "finished": state["current_idx"] >= state["words_per_round"]}

    hint = build_hint(target, attempt, state["detected_word"])
    state["tailored_feedback"] = hint
    state["tailored_timer"]    = time.time()
    return {"result": "wrong", "target": target, "attempt": attempt, "say": hint}


# -----------------------------
# GUI
# -----------------------------
_INTRO_LINE = "Welcome to the Summer Camp! I am Reachy Mini, your word coach."
# On-screen rules (can be wordier — kids/parents can re-read it):
_RULES_LINE = ("We will unscramble animal words! You have 3 chances for each puzzle. "
               "Three-letter words are worth 3 points, four-letter words are worth 5 points. "
               "Arrange the tiles, then press the buzzer!")
# SPOKEN rules — short and crisp for young ears. One clear step at a time,
# no point values (that's on screen). "Ree chee" is intentional: it's how
# Piper pronounces "Reachy" correctly.
_RULES_SAY = ("Here's how we play! Pick a card and put it on the yellow card area. "
              "Then I'll tell you which letters to grab. Let's go!")

# Spoken on the pre-login ready screen. MUST stay byte-for-byte identical to
# speed_game.py's _show_ready_screen() say= string and to the copy in
# reachy_feedback_server.py's pregeneration list — the audio cache is keyed on
# an md5 of the exact text, so all three sharing one string means one cached
# clip serves both games. Change it in one place and you change it in three.
_SEATED_SAY = ("Hello! When you're seated and ready, press the buzzer. "
               "Then we'll get started.")

# Spoken when the scan screen comes up. Also mirrored in the server (its
# face_greet fallback for the "login" phase, and its pre-cache list).
_SCAN_SAY = "Player one! Show me your name card to play!"


class ScrambleGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scramble Game")
        self.root.geometry("1280x780")
        self.root.configure(bg=P['bg'])
        self.root.minsize(1000, 660)

        self._current_level_idx  = 0
        self._selected_category  = LEVELS[0]['category']
        self.state               = new_state(self._selected_category)
        self.reachy              = ReachyFeedbackClient()
        self._logger             = None

        # One video recording per game: created in _start_game (when players
        # are detected), finalized the moment the session ends (via _tick).
        # segment_minutes=0 => a single file per game, not time-based chunks.
        self._recorder           = None
        self._rec_idx            = None   # recording-camera index, resolved once

        self._lvl_btns           = []
        self._cam_stop           = threading.Event()
        self._last_scanned_ids   = []
        self._play_area_hull     = None
        self._card_area_hull     = None
        self._templates          = None
        self._marker_id_dict     = None

        # Stable-word tracking (for invalid-letter detection)
        self._stable_word        = ''
        self._stable_count       = 0
        self._last_gaze          = 0.0
        self._last_observing     = 0.0
        self._tiles_correct_since = None   # when the tiles first spelled it
        # Frame-rate reading stability, maintained by the camera loop. Kept
        # separate from _stable_word/_stable_count, which _tick maintains at
        # TICK rate for the wrong_letter nudge — different rate, different job.
        self._frame_read         = ""     # last camera reading
        self._frame_stable       = 0      # consecutive frames it has held
        self._hands_wait_active  = False  # currently waiting for hands to clear
        self._hands_wait_ready_at = 0.0
        self._hands_wait_deadline = 0.0
        self._last_wrong_letter  = 0.0     # last SPOKEN bad-letter nudge
        self._last_idle_nudge    = 0.0
        self._last_buzzer_reminder = 0.0
        self._last_wrong_letter  = 0.0
        self._tiles_correct_since = None
        self._last_card_nudge    = 0.0
        self._between_words_sent = False
        self._card_stable_word   = None
        self._card_stable_count  = 0
        self._announced_players  = set()   # IDs already announced during login
        self._auto_start_at      = None    # time.time() deadline for auto-start
        self._auto_start_ids     = []      # player set the countdown was armed for
        self._kiosk_reset_token  = 0       # invalidates stale scheduled resets
        self._asked_second       = False   # second-player invite sent this session?

        # ── Pre-login ready screen ───────────────────────────────────────────
        # _awaiting_buzzer is a sub-phase INSIDE waiting_for_ready, not a
        # replacement for it: the camera loop must stay in scan mode so the
        # preview keeps drawing, but _update_scan_labels() must not register
        # or auto-start anyone until the buzzer is pressed.
        self._awaiting_buzzer    = False

        # ── Robot-camera IDs (dual-camera login): {id: last_seen_time} ──────
        self._robot_ids          = {}
        self._robot_ids_lock     = threading.Lock()
        self._robot_face_ago     = None          # secs since robot saw a face
        self._robot_face_time    = 0.0           # local time of that reading

        # ── Inactivity watchdog ──────────────────────────────────────────────
        self._last_activity      = time.time()   # tiles/cards/buzzer/IDs seen

        self.root.bind('<space>', lambda _: self._on_space())
        self.root.bind('<h>',     lambda _: self._hint())
        self.root.bind('<H>',     lambda _: self._hint())
        self.root.bind('<r>',     lambda _: self._new_game())
        self.root.bind('<R>',     lambda _: self._new_game())
        self.root.bind('<c>',     lambda _: self._next_category())
        self.root.bind('<C>',     lambda _: self._next_category())
        self.root.bind('<q>',     lambda _: self._quit())
        self.root.bind('<Q>',     lambda _: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        threading.Thread(target=self._load_templates,  daemon=True).start()
        threading.Thread(target=self._camera_loop,     daemon=True).start()
        threading.Thread(target=self._robot_id_poller, daemon=True).start()
        threading.Thread(target=self._speech_log_sync_loop, daemon=True).start()

        self._build_chrome()
        self._show_ready_screen()
        self._poll_buzzer()
        self._tick()

        # Tell Reachy which game is running so it can give the right welcome
        self.root.after(500, lambda: self.reachy.send("game_start", mode="scramble"))

    def _quit(self):
        # Guard the logger save: if it ever raises, we must still reach
        # root.destroy() below, otherwise mainloop() never returns and the
        # recorder.stop() in main()'s `finally` block never runs either —
        # silently losing the whole session's video, not just the log.
        if self._logger:
            try:
                self._logger.save(finished=False)
            except Exception as e:
                print(f"[LOG] WARN: save-on-quit failed ({e}); "
                      f"continuing shutdown so the recording still finalizes.")
        self._stop_recording()   # finalize this game's video before teardown
        self._sync_speech_log_once()   # final mirror of the robot transcript
        self._cam_stop.set()
        self.root.destroy()

    # ── Templates ─────────────────────────────────────────────────────────────

    def _load_templates(self):
        try:
            t = load_templates()
            d = create_marker_id_dict()
            self._templates      = t
            self._marker_id_dict = d
            build_word_card_map(CATEGORIES)   # register ID→word map for all levels
            self.root.after(0, lambda: self._lbl_status.config(
                text="Templates ready — show a word card or arrange tiles and press SPACE"))
        except Exception as e:
            self.root.after(0, lambda msg=str(e): self._lbl_status.config(
                text=f"Templates not loaded ({msg})"))

    # ── Robot camera: player-ID / presence polling ────────────────────────────

    def _robot_id_poller(self):
        """Poll the Reachy server's /player_ids endpoint.

        Fast during the scan screen so ID cards shown to the ROBOT's camera
        log in just as quickly as ones shown to the game camera; slower (1 s)
        during play, where we only need the robot's face-presence signal for
        the inactivity watchdog.
        """
        while not self._cam_stop.is_set():
            scanning = bool(self.state.get("waiting_for_ready"))
            info = self.reachy.get_presence()
            now  = time.time()
            if info["ids"]:
                with self._robot_ids_lock:
                    for pid in info["ids"]:
                        self._robot_ids[pid] = now
            self._robot_face_ago  = info["face_seen_ago"]
            self._robot_face_time = now
            time.sleep(ROBOT_ID_POLL_S if scanning else 1.0)

    def _current_robot_ids(self):
        """IDs the robot camera has seen within the last ROBOT_ID_TTL_S."""
        cutoff = time.time() - ROBOT_ID_TTL_S
        with self._robot_ids_lock:
            return [pid for pid, t in self._robot_ids.items() if t >= cutoff]

    def _robot_sees_face_within(self, window_s):
        """True if the ROBOT's camera saw a face within the last window_s.

        Unknown (robot offline / never saw a face) counts as False so the
        watchdog falls back to game-camera activity alone.
        """
        ago = self._robot_face_ago
        if ago is None:
            return False
        ago += max(0.0, time.time() - self._robot_face_time)
        return ago <= window_s

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self):
        from camera_select import detection_index
        cam_index = detection_index()
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.root.after(0, lambda: self._lbl_status.config(
                text="No camera — check permissions"))
            return
        # Capture at a native 16:9 mode. The C920 uses its FULL sensor width
        # at 1280x720 (widest field of view); smaller / non-native sizes like
        # 900x560 make the driver snap to a cropped mode, which looks "zoomed
        # in". If it's still too tight, uncomment the CAP_PROP_ZOOM reset below
        # (100 = no digital zoom on the C920).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        # cap.set(cv2.CAP_PROP_ZOOM, 100)
        # Ask the driver to keep only the newest frame. The camera pushes ~30
        # fps but a detection pass takes much longer than 33 ms, so without
        # this the driver queues the frames we're too slow to collect and
        # cap.read() hands us older and older images — the lag GROWS the
        # longer a turn runs. DirectShow honours this inconsistently, hence
        # the explicit drain below as well.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            cap.read()

        while not self._cam_stop.is_set():
            # Drop whatever the driver buffered while we were detecting, so
            # `frame` is the CURRENT view of the table rather than a stale one.
            for _ in range(CAM_DRAIN_FRAMES):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue

            if self.state.get("waiting_for_ready"):
                # ── Scan phase ─────────────────────────────────────────────
                # Local (external) camera detection…
                ids = detect_player_ids(frame)
                # …merged with IDs the ROBOT's camera has seen recently.
                robot_ids = self._current_robot_ids()
                merged = ids + [p for p in robot_ids if p not in ids]
                self._last_scanned_ids = merged
                if merged:
                    self._last_activity = time.time()
                disp = draw_player_markers(frame.copy(), ids)
                # Boundary + debug-overlay detection are cosmetic on the scan
                # screen — run them every 3rd frame so the player-ID pass
                # (the one that matters for login speed) gets more frames/sec.
                self._scan_tick = getattr(self, '_scan_tick', 0) + 1
                if self._scan_tick % 3 == 0:
                    bnd_pts, hull = detect_play_area(frame)
                    if hull is not None:
                        self._play_area_hull = hull
                    self._scan_all_ids = get_all_aruco_ids(frame)
                    self._scan_bnd_pts = bnd_pts
                draw_play_area(disp, self._play_area_hull,
                               getattr(self, '_scan_bnd_pts', []),
                               getattr(self, '_scan_all_ids', None))
                self.root.after(0, lambda f=disp, d=merged: self._on_scan_frame(f, d))

            else:
                # ── Game phase ─────────────────────────────────────────────
                disp     = frame.copy()
                detected = ''

                # ── Card-area hull (4x4 markers 14-17) ─────────────────────
                # Built the same way as the play-area hull: only updated when
                # all 4 corners are visible, so a momentary occlusion can't
                # shrink/skew the zone; otherwise keep the last known-good one.
                _card_pts, _card_hull = detect_card_area(frame)
                if _card_hull is not None:
                    self._card_area_hull = _card_hull

                # ── Word card detection — ONLY inside the card area ────────
                card_word = detect_word_card(frame, card_area_hull=self._card_area_hull)

                prev = self.state.get("card_word")
                if card_word and card_word != prev:
                    # New (or swapped) card detected in the card area —
                    # increment the stability counter. This runs whether or
                    # not we're "waiting_for_card" so a player can also swap
                    # in a different card mid-round; _on_card_detected below
                    # decides what a change means depending on game state.
                    if card_word == getattr(self, "_card_stable_word", None):
                        self._card_stable_count = getattr(self, "_card_stable_count", 0) + 1
                    else:
                        self._card_stable_word  = card_word
                        self._card_stable_count = 1
                    # Only act after 8 stable frames (avoids hold-over false triggers)
                    if self._card_stable_count == 8:
                        self.state["card_word"] = card_word
                        self.root.after(0, lambda w=card_word: self._on_card_detected(w))
                elif not card_word:
                    self._card_stable_word  = None
                    self._card_stable_count = 0
                    # Card area is empty — the kid lifted the finished card.
                    # From here on, putting it back IS a genuine repeat and
                    # earns the real warning.
                    if self.state.get("awaiting_card_removal"):
                        self.state["awaiting_card_removal"] = None
                    # Clear card_word when it's lifted from the area, but only
                    # while we're between words (waiting_for_card) — mid-round
                    # we keep showing the accepted card's word on the HUD even
                    # if the player's hand passes over it briefly.
                    if self.state.get("waiting_for_card") and self.state.get("card_word"):
                        self.state["card_word"] = None
                        def _clear_card():
                            cs = getattr(self, '_lbl_card_status', None)
                            if cs and cs.winfo_exists():
                                cs.config(text="Solving...", fg=P['dim'])
                            self._update_debug_labels()
                        self.root.after(0, _clear_card)

                draw_word_card_overlay(disp, card_word)
                draw_card_area(disp, self._card_area_hull, _card_pts)

                if self._templates is not None and not self.state.get("await_clear"):
                    # Build the play-area hull FIRST from the REAL 4x4 ArUco
                    # boundary markers (IDs 10-13). Only update when all 4
                    # corners are visible so a momentary occlusion can't corrupt
                    # the play area; otherwise keep the last known-good hull.
                    _bnd_pts, _hull = detect_play_area(frame)  # 4x4 only, require all 4
                    if _hull is not None:
                        self._play_area_hull = _hull

                    # Pass the hull into letter detection so any square OUTSIDE
                    # the boundary (e.g. a word card held up to the camera) is
                    # never even classified as a letter.
                    markers, _bnd_guess = detect_markers(
                        frame, self._templates, self._marker_id_dict,
                        allowed_labels=self.state["round_letters"],
                        play_area_hull=self._play_area_hull)

                    # Belt-and-suspenders: also drop any tile whose centre is
                    # outside the hull (handles the first frame before a hull
                    # exists, when play_area_hull was still None above).
                    markers = filter_markers_by_play_area(markers, self._play_area_hull)

                    valid = []
                    for m in markers:
                        corners = m["corners"].astype(int)
                        color   = (0, 220, 80) if m["valid"] else (60, 60, 200)
                        cv2.drawContours(disp, [corners], -1, color, 2)
                        if m["label"]:
                            cv2.putText(disp, m["label"],
                                        tuple(corners.min(axis=0) + [0, -10]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                                        cv2.LINE_AA)
                        if m["valid"] and m["label"]:
                            valid.append((float(corners[:, 0].mean()), m["label"]))

                    valid.sort(key=lambda x: x[0])
                    letters = [lbl for _, lbl in valid]
                    if REVERSE_READING:
                        letters.reverse()
                    detected = "".join(letters)

                    # Gaze event
                    if valid:
                        now = time.time()
                        if now - self._last_gaze > 0.7:
                            self._last_gaze = now
                            mean_x = sum(x for x, _ in valid) / len(valid)
                            self.reachy.send("gaze",
                                             x=round(mean_x / frame.shape[1], 3))

                    # Play-area overlay — reuse the boundary points/hull we
                    # already computed above (don't re-run detection).
                    draw_play_area(disp, self._play_area_hull,
                                   _bnd_pts, get_all_aruco_ids(frame))

                # Tiles removed → lift clear-guard
                if self.state.get("await_clear") and not detected:
                    self.state["await_clear"]   = False
                    self.state["last_detected"] = ""

                if detected == self._frame_read:
                    self._frame_stable += 1
                else:
                    self._frame_read   = detected
                    self._frame_stable = 1

                self.state["detected_word"] = detected
                if detected or card_word:
                    self._last_activity = time.time()
                self.root.after(0,
                    lambda f=disp, w=detected: self._on_game_frame(f, w))

        cap.release()

    # ── Frame handlers ────────────────────────────────────────────────────────

    def _on_scan_frame(self, frame, ids):
        if not self.state.get("waiting_for_ready"):
            return
        self._draw_on_canvas(getattr(self, '_scan_canvas', None), frame)
        self._update_scan_labels(ids)

    def _on_game_frame(self, frame, detected):
        if self.state.get("waiting_for_ready"):
            return
        self._draw_on_canvas(getattr(self, '_cam_canvas', None), frame)
        self._update_detected_label(detected)

    def _draw_on_canvas(self, canvas, frame):
        if not _PIL_OK or canvas is None:
            return
        try:
            if not canvas.winfo_exists():
                return
            w = max(canvas.winfo_width(),  100)
            h = max(canvas.winfo_height(), 60)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo
        except Exception:
            pass

    # ── Persistent chrome ─────────────────────────────────────────────────────

    def _build_chrome(self):
        hdr = tk.Frame(self.root, bg=P['panel'], height=54)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="SCRAMBLE", font=("Helvetica", 17, "bold"),
                 bg=P['panel'], fg=P['green']).pack(side=tk.LEFT, padx=20)

        self._lbl_players = tk.Label(hdr, text="Waiting for players...",
                                     font=("Helvetica", 10),
                                     bg=P['panel'], fg=P['dim'])
        self._lbl_players.pack(side=tk.LEFT, padx=16)

        self._lbl_timer = tk.Label(hdr, text="",
                                   font=("Helvetica", 11, "bold"),
                                   bg=P['panel'], fg=P['dim'])
        self._lbl_timer.pack(side=tk.RIGHT, padx=8)

        self._lbl_score = tk.Label(hdr, text="",
                                   font=("Helvetica", 13, "bold"),
                                   bg=P['panel'], fg=P['gold'])
        self._lbl_score.pack(side=tk.RIGHT, padx=20)

        self._lbl_cat = tk.Label(hdr, text="",
                                 font=("Helvetica", 11),
                                 bg=P['panel'], fg=P['dim'])
        self._lbl_cat.pack(side=tk.RIGHT, padx=8)

        bar = tk.Frame(self.root, bg=P['bg'], pady=6)
        bar.pack(fill=tk.X, padx=12)

        tk.Label(bar, text="LEVEL", font=("Helvetica", 8, "bold"),
                 bg=P['bg'], fg=P['dim']).pack(side=tk.LEFT, padx=(2, 10))

        self._lvl_btns = []
        for i, lv in enumerate(LEVELS):
            btn = tk.Button(
                bar,
                text=f"  LV{i+1}  {lv['name']}  ",
                command=lambda idx=i: self._set_level(idx),
                font=("Helvetica", 9, "bold"),
                relief=tk.FLAT, cursor="hand2", pady=4,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self._lvl_btns.append(btn)

        for label, cmd, bg in (
            ("Hint",     self._hint,         '#166534'),
            ("New Game", self._new_game,     '#1d4ed8'),
        ):
            tk.Button(bar, text=label, command=cmd,
                      bg=bg, fg='white', font=("Helvetica", 9, "bold"),
                      relief=tk.FLAT, padx=14, pady=4, cursor="hand2",
                      activebackground=bg, activeforeground='white',
                      ).pack(side=tk.RIGHT, padx=2)

        self._lbl_status = tk.Label(self.root, text="Starting up...",
                                    font=("Helvetica", 9),
                                    bg=P['bg'], fg=P['dim'])
        self._lbl_status.pack(side=tk.BOTTOM, anchor=tk.W, padx=14, pady=3)

        self._stage = tk.Frame(self.root, bg=P['bg'])
        self._stage.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 0))

        self._update_level_buttons()

    # ── Scan screen ───────────────────────────────────────────────────────────

    def _show_ready_screen(self):
        """Pre-login screen: 'sit down, then press the buzzer'.

        Sits in front of _show_scan_screen(). waiting_for_ready stays True
        (the camera loop needs to remain in scan mode to draw the preview),
        but _awaiting_buzzer suppresses ID registration and auto-start until
        the buzzer is pressed — see _update_scan_labels().
        """
        for w in self._stage.winfo_children():
            w.destroy()

        self._awaiting_buzzer = True

        # Fresh login state — a card flashed at the ready screen must not
        # carry over into the scan screen as an already-registered player.
        self._announced_players = set()
        self._id_sight_counts   = {}
        self._last_scanned_ids  = []
        self._auto_start_at     = None
        self._auto_start_ids    = []
        with self._robot_ids_lock:
            self._robot_ids.clear()

        left = tk.Frame(self._stage, bg='black')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Same attribute name the camera loop's _on_scan_frame() draws to, so
        # the live preview works here with no change to the camera loop.
        self._scan_canvas = tk.Canvas(left, bg='black', highlightthickness=0)
        self._scan_canvas.pack(fill=tk.BOTH, expand=True)

        if not _PIL_OK:
            tk.Label(left, text="pip install pillow  to see camera feed",
                     bg='black', fg='#444', font=("Helvetica", 10)).pack(pady=10)

        right = tk.Frame(self._stage, bg=P['bg'], width=400)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)

        mid = tk.Frame(right, bg=P['bg'])
        mid.pack(expand=True)

        tk.Label(mid, text="SCRAMBLE",
                 font=("Helvetica", 36, "bold"),
                 bg=P['bg'], fg=P['green']).pack(pady=(0, 4))

        tk.Label(mid, text="WORD  UNSCRAMBLING  GAME",
                 font=("Helvetica", 11, "bold"),
                 bg=P['bg'], fg=P['dim']).pack(pady=(0, 26))

        tk.Label(mid, text="Hello!", font=("Helvetica", 32, "bold"),
                 bg=P['bg'], fg=P['gold']).pack(pady=(0, 12))

        tk.Label(mid,
                 text=("When you're seated and ready, press the buzzer. "
                       "Then we'll get started."),
                 font=("Helvetica", 13),
                 bg=P['bg'], fg=P['text'], wraplength=340,
                 justify=tk.CENTER).pack(pady=(0, 20))

        tk.Label(mid, text="Press SPACE or the buzzer when you're ready.",
                 font=("Helvetica", 12, "bold"),
                 bg=P['bg'], fg=P['green']).pack(pady=(0, 30))

        tk.Label(mid, text="After you press the buzzer, I will scan your cards.",
                 font=("Helvetica", 11),
                 bg=P['bg'], fg=P['dim'], wraplength=340,
                 justify=tk.CENTER).pack(pady=(0, 18))

        tk.Button(mid, text="Start scan", command=self._show_scan_screen,
                  bg=P['green'], fg='#000000', font=("Helvetica", 12, "bold"),
                  relief=tk.FLAT, padx=24, pady=10, cursor="hand2",
                  activebackground=P['green'], activeforeground='#000'
                  ).pack(pady=(12, 0))

        self._lbl_status.config(
            text="Press SPACE or the buzzer when you're seated and ready")

        # Tell the robot which screen is up. Silent by design — the robot
        # speaks _SEATED_SAY itself, via face_greet, only once its camera
        # sees someone. An empty table stays quiet.
        self.reachy.send("seated_prompt", say=_SEATED_SAY)

    def _show_scan_screen(self):
        self._awaiting_buzzer = False
        # Flip the robot out of the "ready" phase so face_greet stops offering
        # the buzzer line, and speak the card invite NOW rather than waiting on
        # face_greet: it already fired for this walk-up and is on a 30s
        # cooldown, so relying on it here would leave the kid staring at a
        # silent scan screen right after they pressed the buzzer.
        self.reachy.send("scan_id_prompt", say=_SCAN_SAY)

        # Pre-warm the recording HERE, at the ID-scan screen — several seconds
        # before the first card — so the webcam + ffmpeg startup latency is
        # absorbed during scanning/rules and doesn't clip the start of play.
        # _start_game later calls _start_recording() too, but that's now a
        # no-op while this one is running (force=False), so no restart/clip.
        self._start_recording()
        for w in self._stage.winfo_children():
            w.destroy()

        left = tk.Frame(self._stage, bg='black')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._scan_canvas = tk.Canvas(left, bg='black', highlightthickness=0)
        self._scan_canvas.pack(fill=tk.BOTH, expand=True)

        if not _PIL_OK:
            tk.Label(left, text="pip install pillow  to see camera feed",
                     bg='black', fg='#444', font=("Helvetica", 10)).pack(pady=10)

        right = tk.Frame(self._stage, bg=P['bg'], width=400)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)

        mid = tk.Frame(right, bg=P['bg'])
        mid.pack(expand=True)

        tk.Label(mid, text="SCRAMBLE",
                 font=("Helvetica", 36, "bold"),
                 bg=P['bg'], fg=P['green']).pack(pady=(0, 4))

        tk.Label(mid, text="WORD  UNSCRAMBLING  GAME",
                 font=("Helvetica", 11, "bold"),
                 bg=P['bg'], fg=P['dim']).pack(pady=(0, 22))

        bubble = tk.Frame(mid, bg='#0d1a26')
        bubble.pack(pady=(0, 22), fill=tk.X)
        tk.Frame(bubble, bg='#1e4060', height=2).pack(fill=tk.X)
        self._lbl_robot_msg = tk.Label(
            bubble, text="",
            font=("Helvetica", 11, "italic"),
            bg='#0d1a26', fg=P['blue'],
            wraplength=340, justify=tk.CENTER,
            pady=14, padx=16)
        self._lbl_robot_msg.pack()
        tk.Frame(bubble, bg='#1e4060', height=2).pack(fill=tk.X)

        tk.Frame(mid, bg=P['sep'], height=1).pack(fill=tk.X, pady=(0, 16))

        self._lbl_scan_info = tk.Label(
            mid, text="PLAYER 1 — hold up your name card!",
            font=("Helvetica", 11), bg=P['bg'], fg=P['gold'])
        self._lbl_scan_info.pack()

        self._lbl_scan_detected = tk.Label(
            mid, text="",
            font=("Helvetica", 14, "bold"),
            bg=P['bg'], fg=P['green'])
        self._lbl_scan_detected.pack(pady=10)

        tk.Frame(mid, bg=P['sep'], height=1).pack(fill=tk.X, pady=(0, 14))

        tk.Label(mid, text="Both players show your cards — then we start!",
                 font=("Helvetica", 16, "bold"),
                 bg=P['bg'], fg=P['text']).pack(pady=(0, 4))

        tk.Label(mid, text="(or press SPACE / green buzzer to start right away)",
                 font=("Helvetica", 9), bg=P['bg'], fg=P['dim']).pack()

        tk.Button(mid, text="Skip — play without player ID",
                  command=lambda: self._start_game([]),
                  bg='#2a2a2a', fg=P['dim'],
                  font=("Helvetica", 10),
                  relief=tk.FLAT, padx=14, pady=5, cursor="hand2",
                  ).pack(pady=(16, 0))

        self._lbl_robot_msg.config(text=f'"{_INTRO_LINE}"')
        self._lbl_status.config(
            text="Scan player IDs — press SPACE or buzzer when ready")

        # STANDBY MODE (in-the-wild): the login invite is no longer spoken
        # on a timer. The ROBOT speaks it (via its face_greet logic) only
        # when its camera actually sees someone standing there — otherwise
        # the kiosk stays silent between players. We only update the screen
        # text here.
        self.root.after(4200, self._prompt_scan_login)

    def _prompt_scan_login(self):
        """Update the scan-screen text. SILENT on purpose — the spoken
        login instructions come from the robot itself the moment it spots
        a player (reachy_feedback_server.py's face_greet), so an empty
        room stays quiet."""
        try:
            if self._lbl_robot_msg.winfo_exists():
                self._lbl_robot_msg.config(
                    text='"Player one, please scan your ID card to log in!"')
        except Exception:
            pass

    def _update_scan_labels(self, ids):
        # Ready screen is up — preview only. Registering players here would
        # let a card flashed before the buzzer press log someone in (and, with
        # two cards, auto-start the game) while the screen still says "press
        # the buzzer", which is exactly the confusion the screen exists to
        # prevent.
        if self._awaiting_buzzer:
            return
        try:
            si = getattr(self, '_lbl_scan_info',     None)
            sd = getattr(self, '_lbl_scan_detected', None)
            if sd and sd.winfo_exists():
                if ids:
                    # Register any new players. Login is STICKY: one clean
                    # detection is enough — the kid can put the card down
                    # immediately, the roster never shrinks.
                    self._announce_new_players(ids)
                roster = sorted(self._announced_players)
                if roster:
                    names = " + ".join(get_player_name(p) for p in roster)
                    sd.config(text=f"Logged in:  {names}", fg=P['green'])
                    # Countdown runs on the roster, NOT on what's currently
                    # in frame — flicker or lowering the card can't reset it.
                    self._maybe_auto_start(roster, si)
                else:
                    sd.config(text="Looking for player cards...", fg=P['dim'])
                    if si and si.winfo_exists():
                        si.config(text="PLAYER 1 — hold up your name card!")
        except Exception:
            pass

    def _maybe_auto_start(self, ids, si=None):
        """Start the game automatically once BOTH player IDs are logged in.

        Runs on the Tk main thread (called from _update_scan_labels, which is
        dispatched via root.after). With only ONE player logged in there is
        no countdown — the screen and robot keep asking for player 2; the
        short start countdown only arms once the second card is seen.
        SPACE / green buzzer / Skip can still start a solo game manually.
        """
        if not self.state.get("waiting_for_ready"):
            return
        now = time.time()
        cur = sorted(ids)   # ids == accumulated roster (sticky login)

        # WAIT FOR BOTH PLAYERS: with only one card in, no countdown.
        if len(cur) < 2:
            self._auto_start_at  = None
            self._auto_start_ids = []
            if si and si.winfo_exists() and cur:
                name = get_player_name(cur[0])
                si.config(text=f"Hi {name}!  ✓   Waiting for PLAYER 2 — "
                               "show your card!  (SPACE to play solo)")
            return

        # Both players logged in → arm the short start countdown.
        if cur != sorted(self._auto_start_ids) or self._auto_start_at is None:
            self._auto_start_ids = list(cur)
            self._auto_start_at  = now + AUTO_START_TWO_PLAYERS_S

        remaining = self._auto_start_at - now
        if remaining > 0:
            if si and si.winfo_exists():
                names = " + ".join(get_player_name(p) for p in cur)
                si.config(text=f"Hi {names}!  Starting in "
                               f"{int(remaining) + 1}s…")
            return

        self._auto_start_at  = None
        self._auto_start_ids = []
        print(time.strftime("[%H:%M:%S]"),
              f"auto-start: both players {cur} logged in — starting game")
        self._start_game(list(ids))

    def _announce_new_players(self, ids):
        """Greet each newly-scanned player by name, then — after the FIRST
        player logs in — ask for the second player. The game waits for both;
        when the second card appears they get their own hello and the game
        starts shortly after.
        """
        announced = getattr(self, '_announced_players', set())
        # Confirm each ID across 2 frames (~0.3s) before logging it in —
        # login is sticky/permanent now, so a single misdecoded frame must
        # not be able to register a wrong player.
        counts = getattr(self, '_id_sight_counts', {})
        for p in ids:
            counts[p] = counts.get(p, 0) + 1
        self._id_sight_counts = counts
        new = [p for p in ids if p not in announced and counts.get(p, 0) >= 2]

        # BOTH cards confirmed on the same pass. Handled as one event on
        # purpose: sending player_detected twice meant the second one's
        # clear_queue() (see the server's phase-routing block) wiped the first
        # player's hello before it was spoken, so the pair heard one greeting
        # and then went straight into the rules. `lines` keeps each part
        # separately cacheable — "Hi Alice and Bob!" as a single string would
        # be a name PAIR, which is N^2 and can never be pre-cached.
        if len(new) == 2 and not announced:
            announced.update(new)
            n1, n2 = (get_player_name(x) for x in new)
            self.reachy.send("player_detected",
                             player=new[0], name=f"{n1} + {n2}",
                             lines=[f"Hi {n1}!", f"Hi {n2}!",
                                    "Everyone's here. Let's play!"])
            self._announced_players = announced
            return

        for p in new:
            announced.add(p)
            name = get_player_name(p)
            if len(announced) == 1:
                # FIRST player: greet them, then hand the turn to player two
                # — one spoken line, so nothing can clear-queue the greeting
                # away before it's said.
                say = f"Hi {name}! Now player two — show me your card!"
            elif len(announced) == 2:
                # SECOND player: greet them; the named rules line follows at
                # game start.
                say = f"Hi {name}! Everyone's here. Let's play!"
            else:
                say = f"Hi {name}!"
            self.reachy.send("player_detected", player=p, name=name, say=say)
        self._announced_players = announced

    def _sync_speech_log_once(self):
        """Fetch the robot's transcript and mirror it to this PC's logs/ folder
        (overwrite — the robot is the source of truth). Best-effort."""
        text = self.reachy.fetch_speech_log()
        if text is None:
            return
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "logs")
            os.makedirs(logs_dir, exist_ok=True)
            path = os.path.join(logs_dir,
                                f"reachy_speech_{time.strftime('%Y%m%d')}.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[SPEECH-SYNC] warn: could not write local copy: {e}")

    def _speech_log_sync_loop(self):
        """Background: mirror the robot's spoken-feedback transcript to this PC
        every SPEECH_SYNC_S seconds, so an unattended study collects it with no
        manual scp. Exits when the app shuts down (_cam_stop)."""
        while not self._cam_stop.is_set():
            self._sync_speech_log_once()
            # Sleep in 1s slices so shutdown stays responsive.
            for _ in range(SPEECH_SYNC_S):
                if self._cam_stop.is_set():
                    break
                time.sleep(1)

    def _resolve_rec_idx(self):
        """Resolve the recording camera index once (cached). Uses the same
        name-based lookup as the rest of the app; falls back to 0."""
        if self._rec_idx is not None:
            return self._rec_idx
        try:
            from camera_select import recording_index
            self._rec_idx = recording_index()
        except Exception as e:
            print(f"[REC] could not resolve recording camera ({e}); using 0")
            self._rec_idx = 0
        return self._rec_idx

    def _start_recording(self, force=False):
        """Begin a single-file recording for this game. If one is already
        running, keep it (unless force=True) — so calling this again as a
        safety net can't restart and re-clip an already-warming recording."""
        if SessionRecorder is None:
            return
        if self._recorder is not None and not force:
            return
        self._stop_recording()
        try:
            idx = self._resolve_rec_idx()
            self._recorder = SessionRecorder(cam_index=idx,
                                             segment_minutes=0,
                                             size=RECORD_CAM_SIZE,
                                             fps=RECORD_CAM_FPS).start()
        except Exception as e:
            print(f"[REC] could not start recorder: {e}")
            self._recorder = None

    def _stop_recording(self):
        """Finalize the current recording, if any. Idempotent — safe to call
        from any session-end path (and twice)."""
        rec = self._recorder
        if rec is not None:
            self._recorder = None
            try:
                rec.stop()
            except Exception as e:
                print(f"[REC] error stopping recorder: {e}")

    def _start_game(self, ids):
        self._kiosk_reset_token += 1   # cancel any pending kiosk reset
        self.state["player_ids"]         = ids
        self.state["waiting_for_ready"]  = False
        self.state["waiting_for_card"]   = True   # wait for first card
        self.state["session_start_time"] = time.time()
        self._last_activity              = time.time()   # fresh watchdog window

        if ids:
            names = " + ".join(get_player_name(p) for p in ids)
            self._lbl_players.config(text=names, fg=P['green'])
        else:
            self._lbl_players.config(text="No player scanned", fg=P['dim'])

        self._logger = SessionLogger(self._selected_category, ids)

        # Players are in — start this game's recording. Doing it here (a few
        # seconds before the first card, while the rules are spoken) also gives
        # the encoder headroom so gameplay isn't clipped at the start.
        self._start_recording()

        self.reachy.send("session_start")
        # Spoken rules — ONE FIXED line (no name prefix: the players were
        # greeted by name ~1.5s ago at login). Keeping the text identical
        # every session lets the robot pre-cache the audio; the old
        # "Hi Leo and Zoe! ..." prefix made every name-pair a fresh line
        # that Piper had to synthesize live (~4-6s of silence after login).
        # The server mirrors this exact string in its pre-cache list —
        # if you edit _RULES_SAY, update the copy in
        # reachy_feedback_server.py's _all_lines() too.
        self.reachy.send("ready_prompt", say=_RULES_SAY)

        # Immediately ask for first card — don't wait for 10s nudge timer
        self._last_card_nudge = 0.0  # forces nudge to fire immediately in _tick

        self._show_game_screen()
        cs = getattr(self, '_lbl_card_status', None)
        if cs and cs.winfo_exists():
            cs.config(text="Show a word card to the camera to begin!", fg=P['gold'])
        self._lbl_status.config(text=_RULES_LINE)

    # ── Game screen ───────────────────────────────────────────────────────────

    def _show_game_screen(self):
        for w in self._stage.winfo_children():
            w.destroy()

        # Left panel: status / debug info
        left = tk.Frame(self._stage, bg=P['panel'], width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=4)
        left.pack_propagate(False)

        lhdr = tk.Frame(left, bg=P['panel'])
        lhdr.pack(fill=tk.X, padx=14, pady=(12, 6))

        lv = LEVELS[self._current_level_idx]
        cat_display = self._selected_category.upper().replace('_', ' ')
        tk.Label(lhdr, text=f"LEVEL {self._current_level_idx + 1}",
                 font=("Helvetica", 11, "bold"),
                 bg=P['panel'], fg=lv['color']).pack(side=tk.LEFT)
        tk.Label(lhdr, text=f"  ·  {cat_display}",
                 font=("Helvetica", 11),
                 bg=P['panel'], fg=P['text']).pack(side=tk.LEFT)

        self._lbl_progress = tk.Label(lhdr, text="",
                                      font=("Helvetica", 8),
                                      bg=P['panel'], fg=P['dim'])
        self._lbl_progress.pack(side=tk.RIGHT)

        tk.Frame(left, bg=P['sep'], height=1).pack(fill=tk.X, padx=8, pady=(4, 8))

        # ── Card detection status ─────────────────────────────────────────────
        tk.Label(left, text="WORD CARD",
                 font=("Helvetica", 8, "bold"),
                 bg=P['panel'], fg=P['dim']).pack(anchor=tk.W, padx=14)

        self._lbl_card_status = tk.Label(
            left,
            text="No card detected — show card to camera",
            font=("Helvetica", 11),
            bg=P['panel'], fg=P['dim'],
            wraplength=340, justify=tk.LEFT)
        self._lbl_card_status.pack(anchor=tk.W, padx=14, pady=(2, 12))

        tk.Frame(left, bg=P['sep'], height=1).pack(fill=tk.X, padx=8, pady=(0, 8))

        # ── Word progress (shows ✓ / ✗ / ? without revealing the word) ───────
        tk.Label(left, text="PROGRESS",
                 font=("Helvetica", 8, "bold"),
                 bg=P['panel'], fg=P['dim']).pack(anchor=tk.W, padx=14)

        self._words_frame = tk.Frame(left, bg=P['panel'])
        self._words_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # ── Debug panel ───────────────────────────────────────────────────────
        tk.Frame(left, bg=P['sep'], height=1).pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(left, text="DEBUG",
                 font=("Helvetica", 8, "bold"),
                 bg=P['panel'], fg=P['dim']).pack(anchor=tk.W, padx=14)

        self._lbl_debug_word = tk.Label(
            left, text="Current word: —",
            font=("Helvetica", 9),
            bg=P['panel'], fg='#666',
            wraplength=340, justify=tk.LEFT)
        self._lbl_debug_word.pack(anchor=tk.W, padx=14, pady=(2, 2))

        self._lbl_debug_card = tk.Label(
            left, text="Card ArUco: not detected",
            font=("Helvetica", 9),
            bg=P['panel'], fg='#666',
            wraplength=340, justify=tk.LEFT)
        self._lbl_debug_card.pack(anchor=tk.W, padx=14, pady=(0, 2))

        self._lbl_debug_streak = tk.Label(
            left, text="Streak: 0  |  Wrong streak: 0",
            font=("Helvetica", 9),
            bg=P['panel'], fg='#666')
        self._lbl_debug_streak.pack(anchor=tk.W, padx=14, pady=(0, 8))

        # Right: camera feed + puzzle info
        right = tk.Frame(self._stage, bg=P['panel'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=4)

        cam_wrap = tk.Frame(right, bg='black', height=300)
        cam_wrap.pack(fill=tk.X, padx=8, pady=(8, 0))
        cam_wrap.pack_propagate(False)

        self._cam_canvas = tk.Canvas(cam_wrap, bg='black', highlightthickness=0)
        self._cam_canvas.pack(fill=tk.BOTH, expand=True)

        if not _PIL_OK:
            tk.Label(cam_wrap, text="pip install pillow  to see camera feed",
                     bg='black', fg='#555', font=("Helvetica", 10)).pack(expand=True)

        det_row = tk.Frame(right, bg=P['panel'])
        det_row.pack(fill=tk.X, padx=16, pady=(10, 0))

        tk.Label(det_row, text="CAMERA READS:", font=("Helvetica", 9, "bold"),
                 bg=P['panel'], fg=P['dim']).pack(side=tk.LEFT)

        self._lbl_detected = tk.Label(
            det_row,
            text="— arrange tiles on the table —",
            font=("Courier", 19, "bold"),
            bg=P['panel'], fg=P['dim'])
        self._lbl_detected.pack(side=tk.LEFT, padx=12)

        self._lbl_match = tk.Label(right, text="",
                                   font=("Helvetica", 11, "bold"),
                                   bg=P['panel'], fg=P['green'])
        self._lbl_match.pack()

        tk.Frame(right, bg=P['sep'], height=1).pack(fill=tk.X, padx=16, pady=6)

        info = tk.Frame(right, bg=P['panel'])
        info.pack()

        self._lbl_wordnum = tk.Label(info, text="",
                                     font=("Helvetica", 11),
                                     bg=P['panel'], fg=P['dim'])
        self._lbl_wordnum.pack(pady=(0, 4))

        # Scrambled word — shown to operator/teacher, not kids
        self._lbl_scrambled = tk.Label(info, text="",
                                       font=("Courier", 28, "bold"),
                                       bg=P['panel'], fg='#444')
        self._lbl_scrambled.pack(pady=(0, 2))

        tk.Label(info, text="(scrambled — for operator only)",
                 font=("Helvetica", 8),
                 bg=P['panel'], fg='#333').pack(pady=(0, 4))

        self._lbl_chances = tk.Label(info, text="",
                                     font=("Helvetica", 10),
                                     bg=P['panel'], fg=P['dim'])
        self._lbl_chances.pack()

        tk.Label(info,
                 text="SPACE = submit  ·  H = hint  ·  R = new game  ·  C = category",
                 font=("Helvetica", 9), bg=P['panel'], fg=P['dim']).pack(pady=(4, 0))

        self._lbl_feedback = tk.Label(right, text="",
                                      font=("Helvetica", 14, "bold"),
                                      bg=P['panel'])
        self._lbl_feedback.pack(pady=(10, 2))

        self._lbl_invalid = tk.Label(right, text="",
                                     font=("Helvetica", 10),
                                     bg=P['panel'], fg=P['red'])
        self._lbl_invalid.pack()

        self._lbl_hint = tk.Label(right, text="",
                                  font=("Helvetica", 10),
                                  bg=P['panel'], fg='#c8a42a',
                                  wraplength=560, justify=tk.CENTER)
        self._lbl_hint.pack()

        self._refresh()

    def _update_detected_label(self, detected):
        if self.state.get("waiting_for_ready"):
            return
        try:
            lbl_det = getattr(self, '_lbl_detected', None)
            lbl_mtc = getattr(self, '_lbl_match',    None)
            if not (lbl_det and lbl_det.winfo_exists()):
                return

            if self.state.get("await_clear"):
                lbl_det.config(text="Remove old tiles from the table",
                               fg=P['gold'])
                if lbl_mtc and lbl_mtc.winfo_exists():
                    lbl_mtc.config(text="")
                return

            if detected:
                idx      = self.state["current_idx"]
                is_match = False
                if idx < len(self.state["words"]):
                    anagrams = get_valid_anagrams(self.state["words"][idx][0])
                    is_match = (detected.lower() in anagrams
                                or detected.lower()[::-1] in anagrams)
                lv    = LEVELS[self._current_level_idx]
                color = P['green'] if is_match else lv['color']
                lbl_det.config(text=detected.upper(), fg=color)
                if lbl_mtc and lbl_mtc.winfo_exists():
                    lbl_mtc.config(
                        text="✓  MATCH — press SPACE to confirm!" if is_match else "",
                        fg=P['green'])
            else:
                lbl_det.config(text="— arrange tiles on the table —", fg=P['dim'])
                if lbl_mtc and lbl_mtc.winfo_exists():
                    lbl_mtc.config(text="")
        except Exception:
            pass

    # ── Refresh helpers ───────────────────────────────────────────────────────

    def _refresh(self):
        if self.state.get("waiting_for_ready"):
            return
        lv = LEVELS[self._current_level_idx]
        cat_display = self.state['category'].upper().replace('_', ' ')
        self._lbl_cat.config(
            text=f"LEVEL {self._current_level_idx + 1}  ·  {cat_display}",
            fg=lv['color'])
        self._lbl_score.config(
            text=f"Points: {self.state['score']}")

        prog = getattr(self, '_lbl_progress', None)
        if prog and prog.winfo_exists():
            prog.config(text=f"{sum(self.state['done'])} / {self.state['words_per_round']} done")

        self._update_level_buttons()
        self._refresh_words()
        self._refresh_puzzle()

    def _update_level_buttons(self):
        cur = self._current_level_idx
        for i, (btn, lv) in enumerate(zip(self._lvl_btns, LEVELS)):
            if i == cur:
                btn.config(bg=lv['color'], fg='#000',
                           activebackground=lv['color'], activeforeground='#000')
            else:
                btn.config(bg='#1e1e1e', fg=P['dim'],
                           activebackground='#2a2a2a', activeforeground=P['text'])

    def _refresh_words(self):
        wf = getattr(self, '_words_frame', None)
        if not wf or not wf.winfo_exists():
            return
        for w in wf.winfo_children():
            w.destroy()

        lv_color = LEVELS[self._current_level_idx]['color']
        current  = self.state["current_idx"]

        for i, (target, scrambled) in enumerate(self.state["words"]):
            done    = self.state["done"][i]
            is_cur  = (i == current)
            correct = done and self.state["attempts"][i] < MAX_ATTEMPTS_PER_WORD
            failed  = done and not correct

            bg     = (P['bg_cur']  if is_cur   else
                      P['bg_ok']   if correct  else
                      P['bg_fail'] if failed   else
                      P['card'])
            accent = (lv_color   if is_cur   else
                      P['green'] if correct  else
                      P['red']   if failed   else
                      P['sep'])

            row = tk.Frame(wf, bg=bg, pady=7)
            row.pack(fill=tk.X, pady=1)

            tk.Frame(row, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
            tk.Label(row, text=f"{i+1:02d}",
                     font=("Courier", 12, "bold"),
                     bg=bg, fg=accent, width=3,
                     anchor=tk.CENTER).pack(side=tk.LEFT, padx=(8, 8))

            # Status text — never shows the actual word to kids
            if correct:
                status_text = "✓  correct!"
                status_fg   = P['green']
            elif failed:
                status_text = "✗  out of attempts"
                status_fg   = P['red']
            elif is_cur:
                attempts = self.state["attempts"][i]
                if attempts == 0:
                    status_text = "▶  solving now..."
                else:
                    chances = MAX_ATTEMPTS_PER_WORD - attempts
                    status_text = f"▶  {chances} chance{'s' if chances != 1 else ''} left"
                status_fg = lv_color
            else:
                status_text = "·  waiting"
                status_fg   = P['dim']

            tk.Label(row, text=status_text,
                     font=("Helvetica", 11),
                     bg=bg, fg=status_fg,
                     anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Update debug labels
        self._update_debug_labels()

    def _refresh_puzzle(self):
        lw = getattr(self, '_lbl_wordnum',  None)
        ls = getattr(self, '_lbl_scrambled', None)
        lc = getattr(self, '_lbl_chances',  None)
        if not (lw and lw.winfo_exists()):
            return

        # Waiting for card — no word set yet
        if self.state.get("waiting_for_card") or not self.state["words"]:
            lw.config(text="Show a word card to the camera!", fg=P['gold'])
            if ls and ls.winfo_exists():
                ls.config(text="", fg=P['dim'])
            if lc and lc.winfo_exists():
                lc.config(text="")
            return

        if self.state["current_idx"] >= len(self.state["words"]):
            lw.config(text="GAME  OVER", fg=P['green'])
            if ls and ls.winfo_exists():
                ls.config(text=f"{self.state['score']}  points",
                          fg=P['gold'])
            if lc and lc.winfo_exists():
                lc.config(text="")
            return

        idx = self.state["current_idx"]
        _, scrambled = self.state["words"][idx]
        lv = LEVELS[self._current_level_idx]

        lw.config(text=f"Word  {idx + 1}  of  {self.state['words_per_round']}", fg=P['dim'])
        if ls and ls.winfo_exists():
            ls.config(text="  ".join(scrambled.upper()), fg=lv['color'])

        if lc and lc.winfo_exists():
            chances = max(0, MAX_ATTEMPTS_PER_WORD - self.state["attempts"][idx])
            lc.config(text=f"Chances left: {chances}", fg=P['dim'])

        lh = getattr(self, '_lbl_hint', None)
        if lh and lh.winfo_exists():
            lh.config(text="")

    # ── Periodic tick (100 ms) ────────────────────────────────────────────────

    def _tick(self):
        try:
            now = time.time()

            # One recording per game: finalize the instant the session ends,
            # whatever ended it (timeout, inactivity, red buzzer, level
            # complete all set session_done). _tick runs continuously, so this
            # one spot catches every end path without touching each of them.
            if self.state.get("session_done") and self._recorder is not None:
                self._stop_recording()

            # Session timer
            lt = getattr(self, '_lbl_timer', None)
            if lt and lt.winfo_exists():
                if (not self.state.get("waiting_for_ready")
                        and not self.state.get("session_done")):
                    remaining = get_time_remaining(self.state)
                    lt.config(
                        text=f"{remaining // 60:02d}:{remaining % 60:02d}",
                        fg=P['red'] if remaining < 60 else P['dim'])
                    if remaining <= 0:
                        self.state["session_done"] = True
                        if self._logger:
                            self._logger.save(finished=False)
                        self.reachy.send("session_timeout",
                                         score=sum(self.state["done"]),
                                         total=self.state["words_per_round"])
                        # Goodbye: final score + farewell, after the
                        # session_timeout speech has had time to play.
                        self.root.after(5000,
                                        lambda: self._send_goodbye("timeout"))
                        fb = getattr(self, '_lbl_feedback', None)
                        if fb and fb.winfo_exists():
                            fb.config(text="Time's up!", fg=P['gold'])
                elif self.state.get("session_done"):
                    lt.config(text="00:00", fg=P['red'])

            # ── Standby wake: early kiosk reset on new player ──────────────
            # Between goodbye and the timed kiosk reset, watch the robot's
            # face signal: once the departing players are gone for >=20s,
            # arm; the next face that appears is a NEW player — reset to the
            # scan screen right away so the robot's spoken login invite
            # lands on a screen that can actually log them in.
            if (self.state.get("session_done")
                    and not self.state.get("waiting_for_ready")):
                if not self._robot_sees_face_within(20.0):
                    self._standby_wake_armed = True
                elif (getattr(self, '_standby_wake_armed', False)
                        and self._robot_sees_face_within(2.0)):
                    self._standby_wake_armed = False
                    print(time.strftime("[%H:%M:%S]"),
                          "standby wake: new player spotted — early reset")
                    self._kiosk_reset(self._kiosk_reset_token)

            # ── Inactivity watchdog ─────────────────────────────────────────
            # Active during play. Ends the session only when BOTH signals are
            # quiet: the robot's camera hasn't seen a face AND the game camera
            # has seen no tiles/cards/buzzer for INACTIVITY_TIMEOUT_S.
            if (not self.state.get("waiting_for_ready")
                    and not self.state.get("session_done")):
                quiet_local = now - self._last_activity > INACTIVITY_TIMEOUT_S
                quiet_robot = not self._robot_sees_face_within(INACTIVITY_TIMEOUT_S)
                if quiet_local and quiet_robot:
                    self._end_by_inactivity()

            # Invalid-letter feedback timeout
            if (self.state.get("invalid_feedback")
                    and now - self.state.get("invalid_feedback_timer", 0) > 2.5):
                self.state["invalid_feedback"] = None
                li = getattr(self, '_lbl_invalid', None)
                if li and li.winfo_exists():
                    li.config(text="")

            # Tailored feedback timeout
            if (self.state.get("tailored_feedback")
                    and now - self.state.get("tailored_timer", 0) > 5.0):
                self.state["tailored_feedback"] = None
                lh = getattr(self, '_lbl_hint', None)
                if lh and lh.winfo_exists():
                    lh.config(text="")

            # Stable-word invalid-letter detection
            in_play = (not self.state.get("waiting_for_ready")
                       and not self.state.get("await_clear")
                       and not self.state.get("session_done")
                       and not self.state.get("waiting_for_card")
                       and self.state["current_idx"] < len(self.state["words"]))
            if in_play:
                detected = self.state.get("detected_word", "")
                if detected == self._stable_word:
                    self._stable_count += 1
                else:
                    self._stable_word  = detected
                    self._stable_count = 1

                if (self._stable_word
                        and self._stable_count == STABLE_FRAMES
                        and self._stable_word != self.state.get("last_detected")):
                    self.state["last_detected"] = self._stable_word
                    if check_invalid_letter(self.state):
                        idx     = self.state["current_idx"]
                        allowed = set(self.state["words"][idx][1].upper())
                        extras  = sorted(
                            {ch for ch in self._stable_word.upper()
                             if ch not in allowed})
                        self.state["invalid_feedback"]        = "wrong_letter"
                        self.state["invalid_feedback_timer"]  = now
                        # Screen updates on EVERY bad-letter reading; only the
                        # spoken nudge is rate-limited. Kids swap tiles fast,
                        # and a line per swap both drowns out the verdict and
                        # gets annoying by the third word.
                        li = getattr(self, '_lbl_invalid', None)
                        if li and li.winfo_exists():
                            li.config(text="Is that the right letter?")
                        if now - self._last_wrong_letter > WRONG_LETTER_REPEAT_S:
                            self._last_wrong_letter = now
                            self.reachy.send("wrong_letter", letters=extras)

                # Reset idle nudge timer when a new word starts
                if self.state.pop("_idle_nudge_reset", False):
                    self._hands_wait_active = False   # cancel wait from last word
                    self._last_idle_nudge = 0.0
                    self._last_buzzer_reminder = 0.0
                    self._last_wrong_letter   = 0.0
                    self._tiles_correct_since = None

                # Observing — fires when tiles are on table, every 12s
                if detected and now - self._last_observing > 12.0:
                    self.reachy.send("observing")
                    self._last_observing = now

                # Start the word timer when the kid first puts tiles down
                if detected and self.state.get("word_start_time") is None:
                    self.state["word_start_time"] = now

                idx_now = self.state["current_idx"]
                target_now = self.state["words"][idx_now][0]
                tiles_correct = bool(detected) and detected.upper() == target_now.upper()

                # Buzzer reminder — the tiles ALREADY spell the target word,
                # the kid just hasn't pressed yet. Screen says so immediately;
                # the VOICE waits BUZZER_HINT_DELAY_S, because the first few
                # seconds after the tiles land are exactly when the press
                # happens and anything spoken there collides with the verdict.
                if tiles_correct:
                    if self._tiles_correct_since is None:
                        self._tiles_correct_since = now
                    fb = getattr(self, '_lbl_feedback', None)
                    if fb and fb.winfo_exists():
                        fb.config(text="Looks right! Press the buzzer!",
                                  fg=P['green'])
                    if (now - self._tiles_correct_since > BUZZER_HINT_DELAY_S
                            and now - getattr(self, "_last_buzzer_reminder", 0)
                                > BUZZER_HINT_REPEAT_S):
                        self._last_buzzer_reminder = now
                        self.reachy.send("press_buzzer")
                else:
                    # Tiles disturbed — the kid is working again, so the stall
                    # clock restarts from scratch on the next correct reading.
                    self._tiles_correct_since = None

                # Idle nudge — fires after 15s on a word regardless of tiles
                # so Reachy speaks during thinking gaps between attempts too.
                # Skipped when the tiles are already correct — that case is
                # handled by the buzzer reminder above; sending "idle" here
                # too used to mean a correctly-solved word could still get a
                # canned "try swapping some tiles" line, which makes no
                # sense once the word is already right.
                word_start = self.state.get("word_start_time")
                if (word_start
                        and not tiles_correct
                        and not self.state.get("waiting_for_card")
                        and idx_now < len(self.state["words"])
                        and now - word_start > 15.0
                        and now - getattr(self, "_last_idle_nudge", 0) > 20.0):
                    self._last_idle_nudge = now
                    attempts = self.state["attempts"][idx_now]
                    self.reachy.send("idle",
                                     attempts=attempts,
                                     target=target_now)

            # Waiting for card nudge — fires immediately then every 10s
            if (not self.state.get("waiting_for_ready")
                    and not self.state.get("session_done")
                    and self.state.get("waiting_for_card")):
                if now - getattr(self, "_last_card_nudge", 0) > 10.0:
                    self._last_card_nudge = now
                    self.reachy.send("waiting_for_card")
                    # Update GUI card status
                    cs = getattr(self, '_lbl_card_status', None)
                    if cs and cs.winfo_exists():
                        msg = ("Show your first word card!" if self.state["current_idx"] == 0
                               else f"Word {self.state['current_idx'] + 1} of "
                                    f"{self.state['words_per_round']} — show the next card!")
                        cs.config(text=msg, fg=P['gold'])

        except Exception:
            pass

        self.root.after(100, self._tick)

    # ── Goodbye / inactivity end ──────────────────────────────────────────────

    def _send_goodbye(self, reason):
        """Robot announces the final score and says a warm farewell."""
        ids   = self.state.get("player_ids") or []
        names = " and ".join(get_player_name(p) for p in ids) or "You"
        self.reachy.send("goodbye", mode="scramble",
                         names=names,
                         # "scored X out of {total} words" — the numerator must
                         # be WORDS SOLVED, not state["score"] (which is points,
                         # +3/+5 per word, so it could exceed the word count and
                         # produce nonsense like "34 out of 29").
                         score=sum(self.state.get("done", [])),
                         total=self.state.get("words_per_round", 0),
                         reason=reason)
        # Kiosk mode: every ending funnels through here (finished / timeout /
        # inactivity), so this is the one place to arm the auto-reset that
        # brings the scan screen back for the next player.
        self._schedule_kiosk_reset()

    # ── Kiosk auto-reset (unattended deployment) ─────────────────────────────

    def _schedule_kiosk_reset(self):
        """Arm a return-to-scan-screen reset KIOSK_RESET_DELAY_S from now.

        Token-guarded: any manual New Game / session start bumps the token,
        which invalidates every previously scheduled reset — so a counselor
        (or a kid pressing R) can't have the screen yanked away mid-game by
        a stale timer from the previous session.
        """
        self._kiosk_reset_token += 1
        token = self._kiosk_reset_token
        print(time.strftime("[%H:%M:%S]"),
              f"kiosk: scan screen returns in {KIOSK_RESET_DELAY_S}s")
        self._lbl_status.config(
            text=f"Game over — ready for the next player in "
                 f"{KIOSK_RESET_DELAY_S // 60} minutes (or press R now)")
        self.root.after(int(KIOSK_RESET_DELAY_S * 1000),
                        lambda: self._kiosk_reset(token))

    def _kiosk_reset(self, token):
        """Return to the scan/login screen so the next player can start."""
        if token != self._kiosk_reset_token:
            return   # superseded by a manual New Game / new session
        if self.state.get("waiting_for_ready"):
            return   # already on the scan screen
        print(time.strftime("[%H:%M:%S]"),
              "kiosk: resetting to scan screen for the next player")

        # Close out any logger left open by a non-standard ending.
        if self._logger:
            try:
                self._logger.save(finished=False)
            except Exception:
                pass
            self._logger = None

        # Belt-and-suspenders: make sure no recording is left running while the
        # kiosk sits idle waiting for the next group (normally _tick already
        # stopped it when session_done flipped).
        self._stop_recording()

        # Next player starts fresh at level 1.
        self._current_level_idx = 0
        self._selected_category = LEVELS[0]['category']
        self.state = new_state(self._selected_category)   # waiting_for_ready=True

        # Clear per-session trackers so nothing leaks into the next session.
        self._announced_players  = set()
        self._id_sight_counts    = {}
        self._asked_second       = False
        self._last_scanned_ids   = []
        self._auto_start_at      = None
        self._auto_start_ids     = []
        with self._robot_ids_lock:
            self._robot_ids.clear()
        self._stable_word        = ''
        self._stable_count       = 0
        self._card_stable_word   = None
        self._card_stable_count  = 0
        self._last_idle_nudge    = 0.0
        self._last_buzzer_reminder = 0.0
        self._last_card_nudge    = 0.0
        self._between_words_sent = False
        self._last_activity      = time.time()
        self._standby_wake_armed = False

        lp = getattr(self, '_lbl_players', None)
        if lp and lp.winfo_exists():
            lp.config(text="No player scanned", fg=P['dim'])
        self._update_level_buttons()

        # Rebuild the login UI. We go back to the READY screen, not straight
        # to the scan screen: the next kids need the same "sit down, then
        # buzz" beat the first pair got. _show_ready_screen() clears the login
        # trackers and re-arms the face-gated invite, so an empty room stays
        # silent until somebody actually walks up.
        self._show_ready_screen()
        self._lbl_status.config(
            text="Press SPACE or the buzzer when you're seated and ready")

    def _end_by_inactivity(self):
        """Nobody is around — end the session and say goodbye."""
        if self.state.get("session_done") or self.state.get("waiting_for_ready"):
            return
        print(f"[GAME] Inactivity timeout ({INACTIVITY_TIMEOUT_S}s) — ending session.")
        self.state["session_done"] = True
        if self._logger:
            self._logger.save(finished=False)

        fb = getattr(self, '_lbl_feedback', None)
        if fb and fb.winfo_exists():
            fb.config(text="Game ended — see you next time!", fg=P['gold'])
        self._lbl_status.config(
            text="No activity detected — game ended. Press New Game to play again.")

        self._send_goodbye("inactivity")

    # ── Buzzer polling ────────────────────────────────────────────────────────

    def _poll_buzzer(self):
        if _buzzer_event.is_set():
            _buzzer_event.clear()
            self._on_space()
        if _quit_event.is_set():
            _quit_event.clear()
            self._on_red_buzzer()
        self.root.after(50, self._poll_buzzer)

    def _on_red_buzzer(self):
        """RED buzzer: end the current session gracefully.

        Deliberately NOT self._quit() — a hard app quit would let one curious
        kid kill the unattended kiosk for the day. Instead the session ends
        the same way inactivity does: logger saved, Reachy announces the
        score and says goodbye, and the kiosk reset brings the scan screen
        back for the next player. On the scan screen it does nothing.
        """
        if self.state.get("waiting_for_ready"):
            return   # nothing to end yet
        if self.state.get("session_done"):
            return   # already ended — goodbye/reset in progress
        print(time.strftime("[%H:%M:%S]"), "red buzzer — ending session")
        self.state["session_done"] = True
        if self._logger:
            self._logger.save(finished=False)
        fb = getattr(self, '_lbl_feedback', None)
        if fb and fb.winfo_exists():
            fb.config(text="Game ended — thanks for playing!", fg=P['gold'])
        self._send_goodbye("quit_button")

    # ── Input handlers ────────────────────────────────────────────────────────

    def _update_debug_labels(self):
        """Update the debug panel on the left side of the game screen."""
        idx = self.state["current_idx"]
        n   = len(self.state["words"])

        ld = getattr(self, '_lbl_debug_word', None)
        if ld and ld.winfo_exists():
            if idx < n:
                target, scrambled = self.state["words"][idx]
                ld.config(text=f"Current word: {target.upper()}  (scrambled: {scrambled.upper()})")
            elif self.state.get("waiting_for_card"):
                ld.config(text="Current word: — (waiting for card)")
            else:
                ld.config(text="Current word: — (round complete)")

        lc = getattr(self, '_lbl_debug_card', None)
        if lc and lc.winfo_exists():
            card = self.state.get("card_word")
            if card:
                lc.config(text=f"Card ArUco: ✓ detected → {card.upper()}", fg=P['green'])
            else:
                lc.config(text="Card ArUco: not detected", fg='#666')

        ls = getattr(self, '_lbl_debug_streak', None)
        if ls and ls.winfo_exists():
            streak       = self.state.get("streak", 0)
            wrong_streak = self.state.get("wrong_streak", 0)
            ls.config(text=f"Streak: {streak}  |  Wrong streak: {wrong_streak}")

    def _warn_already_played(self, word: str):
        """Shared 'you already did this one' nudge for both the fresh-card
        and mid-round card-swap paths."""
        if word != getattr(self, "_last_repeat_warned", None):
            self._last_repeat_warned = word
            self.reachy.send("card_repeated", word=word)
            cs = getattr(self, '_lbl_card_status', None)
            if cs and cs.winfo_exists():
                cs.config(
                    text=f"⚠ Already played: {word.upper()} — try a different card!",
                    fg=P['gold'])

    def _on_card_detected(self, word: str):
        """Called when the card sitting in the card area changes — either a
        fresh card while we're waiting for one, or a different card swapped
        in mid-round (the player pulled their card and put down a new one)."""
        if self.state.get("waiting_for_ready") or self.state.get("session_done"):
            return

        # Just finished a word — the old card is still on the table and the
        # robot is still saying "bring a new card". Ignore the area entirely
        # for a beat rather than reacting to what's already there.
        if time.time() < self.state.get("card_grace_until", 0):
            return

        if self.state.get("waiting_for_card"):
            # ── Fresh card: this starts the next word in the round ─────────
            if word.lower() in self.state.get("used_words", set()):
                # The word we JUST played doesn't count as a repeat until its
                # card has actually been lifted out of the area — see
                # advance_to_next_word(). Silent: no scolding, no screen warning.
                if word.lower() == self.state.get("awaiting_card_removal"):
                    return
                self._warn_already_played(word)
                return

            self._last_repeat_warned = None

            # Add the word to the round
            if not add_card_word(self.state, word):
                return

            # Reset timers and card stable tracking for fresh word
            self._last_idle_nudge    = 0.0
            self._last_buzzer_reminder = 0.0
            self._last_observing     = 0.0
            self._card_stable_word   = None
            self._card_stable_count  = 0

            self.reachy.send("card_detected", word=word)

            cs = getattr(self, '_lbl_card_status', None)
            if cs and cs.winfo_exists():
                cs.config(
                    text=f"✓ Card: {word.upper()} — arrange tiles then press buzzer!",
                    fg=P['green'])

            fb = getattr(self, '_lbl_feedback', None)
            if fb and fb.winfo_exists():
                fb.config(text=f"Card: {word.upper()} — arrange tiles then buzz!",
                          fg=P['blue'])

            self._refresh()
            self._update_debug_labels()
            return

        # ── Not waiting for a card: the player swapped to a different card
        #    mid-round. Abandon the word currently in progress and start
        #    over with whatever new card they brought. ────────────────────
        idx = self.state["current_idx"]
        if idx >= len(self.state["words"]):
            return   # round already finished — nothing to swap into

        if word.lower() in self.state.get("used_words", set()):
            self._warn_already_played(word)
            return

        self._last_repeat_warned = None
        self._swap_current_word(word)

    def _swap_current_word(self, word: str):
        """Replace the word currently in progress with `word`, because the
        player brought a different card to the card area mid-round. Resets
        that round slot (attempts, tiles-seen, feedback) as if it were being
        started fresh, and requires the old letter tiles to be cleared
        before the new word can be checked."""
        word = word.lower()
        idx  = self.state["current_idx"]

        old_word, _ = self.state["words"][idx]
        self.state["used_words"].discard(old_word)
        self.state["used_words"].add(word)
        self.state["words"][idx] = (word, PRINTED_SCRAMBLES.get(word, scramble(word)))
        self.state["done"][idx]                  = False
        self.state["attempts"][idx]              = 0
        self.state["seen_letters_for_word"][idx] = False
        self.state["round_letters"]   = set("".join(w for w, _ in self.state["words"]).upper())
        self.state["detected_word"]   = ""
        self.state["last_detected"]   = ""
        self.state["feedback"]        = None
        self.state["feedback_timer"]  = 0
        self.state["invalid_feedback"]      = None
        self.state["invalid_feedback_timer"] = 0
        self.state["tailored_feedback"] = None
        self.state["tailored_timer"]  = 0
        self.state["await_clear"]     = True   # make them clear the old tiles first
        self.state["word_start_time"] = None

        self._last_idle_nudge   = 0.0
        self._last_buzzer_reminder = 0.0
        self._last_observing    = 0.0
        self._card_stable_word  = None
        self._card_stable_count = 0

        self.reachy.send("card_changed", word=word)

        cs = getattr(self, '_lbl_card_status', None)
        if cs and cs.winfo_exists():
            cs.config(
                text=f"✓ New card: {word.upper()} — clear old tiles, then arrange & buzz!",
                fg=P['green'])

        fb = getattr(self, '_lbl_feedback', None)
        if fb and fb.winfo_exists():
            fb.config(text=f"Card changed: {word.upper()} — clear tiles, then arrange & buzz!",
                      fg=P['blue'])

        self._refresh()
        self._update_debug_labels()

    def _on_space(self):
        self._last_activity = time.time()
        # Debounce — ignore presses within 2 seconds of the last one
        # to prevent kids (or testers) spamming through words
        now = time.time()
        if now - getattr(self, '_last_space', 0) < 2.0:
            return
        self._last_space = now

        if self._awaiting_buzzer:
            # Ready screen → scan screen. Checked BEFORE waiting_for_ready,
            # which is also True here; without this the first press would
            # start a game with nobody logged in.
            self._show_scan_screen()
        elif self.state.get("waiting_for_ready"):
            self._start_game(list(self._last_scanned_ids))
        else:
            self._buzzer_check()

    def _buzzer_check(self):
        """Judge the word, but only once the kid's hands are off the tiles.

        FAST — if the reading was already rock-steady at the moment of the
        press, the tiles were untouched, so judge immediately with no delay.
        This is the common case for a kid who lays the word out, sits back,
        then presses.

        SLOW — anything else waits HANDS_DELAY_S first, because a hand still
        over the tiles reads as a stable-but-partial word and can't be
        trusted. After the delay, the first stable reading is judged (up to
        HANDS_WAIT_MAX_S more); on timeout the current reading is judged
        as-is so a press ALWAYS gets an answer.
        """
        if self._hands_wait_active:
            return   # already reading a press — ignore re-presses

        # Cheap rejects first: these don't involve reading tiles at all, so
        # they must answer instantly rather than after a 2s stall.
        if (self.state.get("session_done")
                or self.state.get("waiting_for_card")
                or self.state.get("await_clear")
                or self.state["current_idx"] >= len(self.state["words"])):
            self._check()
            return

        # ── FAST PATH ────────────────────────────────────────────────────
        if (self.state.get("detected_word", "")
                and self._frame_stable >= HANDS_FAST_STABLE_FRAMES):
            print(f"[GAME] buzzer pressed — reading already stable for "
                  f"{self._frame_stable} frames; judging now")
            self._check()
            return

        # ── SLOW PATH ────────────────────────────────────────────────────
        self._hands_wait_active   = True
        self._hands_wait_ready_at = time.time() + HANDS_DELAY_S
        self._hands_wait_deadline = self._hands_wait_ready_at + HANDS_WAIT_MAX_S
        fb = getattr(self, '_lbl_feedback', None)
        if fb and fb.winfo_exists():
            fb.config(text="Got it! Hands away — reading your word…",
                      fg=P['gold'])
        print("[GAME] buzzer pressed — reading tiles in "
              f"{HANDS_DELAY_S:.0f}s (hands-away delay)")
        self.root.after(HANDS_POLL_MS, self._poll_hands_wait)

    def _poll_hands_wait(self):
        if not self._hands_wait_active:
            return
        # The session may have ended (timeout / kiosk reset) while we waited.
        if (self.state.get("session_done")
                or self.state.get("waiting_for_ready")):
            self._hands_wait_active = False
            return
        now = time.time()
        if now < self._hands_wait_ready_at:
            self.root.after(HANDS_POLL_MS, self._poll_hands_wait)
            return   # still inside the fixed post-press delay
        detected  = self.state.get("detected_word", "")
        stable_ok = bool(detected) and self._frame_stable >= HANDS_STABLE_FRAMES
        timed_out = now >= self._hands_wait_deadline
        if stable_ok or timed_out:
            self._hands_wait_active = False
            if timed_out and not stable_ok:
                print("[GAME] hands-wait timed out — judging current reading as-is")
            if self.state.get("detected_word", ""):
                self._check()
            else:
                fb = getattr(self, '_lbl_feedback', None)
                if fb and fb.winfo_exists():
                    fb.config(text="I don't see any tiles!", fg=P['red'])
            return
        self.root.after(HANDS_POLL_MS, self._poll_hands_wait)

    def _check(self):
        if self.state.get("session_done"):
            return
        if self.state.get("waiting_for_card"):
            fb = getattr(self, '_lbl_feedback', None)
            if fb and fb.winfo_exists():
                fb.config(text="Show a word card to the camera first!",
                          fg=P['gold'])
            return
        if self.state["current_idx"] >= len(self.state["words"]):
            return
        if self.state.get("await_clear"):
            self._lbl_status.config(
                text="Remove old tiles from the table first!")
            return

        pre_idx = self.state["current_idx"]
        result  = check_word(self.state)

        # Log result
        if result["result"] in ("correct", "wrong", "failed") and self._logger:
            status        = "incorrect" if result["result"] == "wrong" else result["result"]
            attempts_used = self.state["attempts"][pre_idx]
            self._logger.record_puzzle(result["target"], status, attempts_used)
            if result.get("finished"):
                self._logger.save(finished=True)

        if result["result"] == "correct":
            points = result.get("points", 0)
            self._lbl_feedback.config(text=f"CORRECT!  +{points} points", fg=P['green'])
            clue   = WORD_CLUES.get(result["target"].lower(), "")
            streak = result.get("streak", 0)
            score  = result.get("score", 0)
            total  = result.get("total", self.state["words_per_round"])
            if result.get("finished"):
                self._on_level_complete()
            else:
                self.reachy.send("correct", target=result["target"],
                                 clue=clue, streak=streak, points=points,
                                 score=score, total=total, next=True)
                # Fire "show next card" nudge after 4s (after correct speech finishes)
                self._last_card_nudge = time.time() - 6.0

        elif result["result"] == "wrong":
            self._lbl_feedback.config(text="Not quite — keep trying!", fg=P['red'])
            self.reachy.send("wrong", target=result["target"],
                             attempt=result["attempt"], say=result["say"])

        elif result["result"] == "failed":
            self._lbl_feedback.config(
                text=f"The word was {result['target'].upper()}", fg=P['red'])
            clue         = WORD_CLUES.get(result["target"].lower(), "")
            wrong_streak = result.get("wrong_streak", 0)
            if result.get("finished"):
                self._on_level_complete()
            else:
                self.reachy.send("word_failed",
                                 target=result["target"], clue=clue,
                                 wrong_streak=wrong_streak,
                                 say=result["say"])
                # Fire "show next card" nudge after 5s (after reveal speech finishes)
                self._last_card_nudge = time.time() - 5.0

        tailored = self.state.get("tailored_feedback") or ""
        lh = getattr(self, '_lbl_hint', None)
        if lh and lh.winfo_exists():
            lh.config(text=tailored, fg='#c8a42a')

        print(time.strftime("[%H:%M:%S]"), "space pressed, result:", result["result"])
        self._refresh()

    def _on_level_complete(self):
        score    = self.state["score"]
        total    = self.state["words_per_round"]
        next_idx = self._current_level_idx + 1
        is_last  = next_idx >= len(LEVELS)

        # Send WORDS SOLVED (not points) — the robot's summary says "solving
        # X of Y words", so points here would misreport it. The on-screen
        # label below still shows points, which is a separate metric.
        self.reachy.send("level_complete",
                         score=sum(self.state["done"]), total=total)

        fb = getattr(self, '_lbl_feedback', None)
        if fb and fb.winfo_exists():
            if is_last:
                # Terminal state: stops the session timer and the inactivity
                # watchdog, so the kiosk reset armed by the goodbye below is
                # the single, final countdown to the next player's session.
                self.state["session_done"] = True
                fb.config(
                    text=f"🎉 All done!  {score} points  —  Amazing work!",
                    fg=P['gold'])
                # Whole game finished → score announcement + farewell,
                # after the level_complete speech has had time to play.
                self.root.after(5000,
                                lambda: self._send_goodbye("finished"))
            else:
                next_lv = LEVELS[next_idx]
                fb.config(
                    text=f"Level {self._current_level_idx + 1} complete! "
                         f"Moving to {next_lv['name']} in 3 seconds…",
                    fg=P['green'])
                self.root.after(3000, lambda: self._advance_level(next_idx))

    def _advance_level(self, idx):
        if idx >= len(LEVELS):
            return
        old_ids = self.state["player_ids"]
        if self._logger:
            self._logger.save(finished=True)

        self._current_level_idx  = idx
        self._selected_category  = LEVELS[idx]['category']
        self.state               = new_state(self._selected_category)
        self.state["player_ids"]         = old_ids
        self.state["waiting_for_ready"]  = False
        self.state["waiting_for_card"]   = True   # wait for next card
        self.state["session_start_time"] = time.time()
        self._logger             = SessionLogger(self._selected_category, old_ids)
        self._stable_word        = ''
        self._stable_count       = 0
        self._last_idle_nudge    = 0.0
        self._last_buzzer_reminder = 0.0
        self._last_card_nudge    = 0.0
        self._between_words_sent = False

        lv = LEVELS[idx]
        self.reachy.send("level_up", category=self._selected_category)
        self._lbl_status.config(
            text=f"Level {idx + 1}: {lv['desc']}  —  show a word card!")

        self._show_game_screen()
        self._update_level_buttons()

    def _hint(self):
        if (self.state.get("waiting_for_ready")
                or self.state.get("session_done")
                or self.state.get("waiting_for_card")
                or self.state["current_idx"] >= len(self.state["words"])):
            return
        idx    = self.state["current_idx"]
        target = self.state["words"][idx][0]
        level  = min(MAX_ATTEMPTS_PER_WORD, self.state["attempts"][idx] + 1)
        hint   = build_hint(target, level, self.state["detected_word"])
        self.state["tailored_feedback"] = hint
        self.state["tailored_timer"]    = time.time()
        say = hint if hint.startswith("Here is a clue") else "Here is a hint. " + hint
        self.reachy.send("hint", target=target, hint_level=level, say=say)
        lh = getattr(self, '_lbl_hint', None)
        if lh and lh.winfo_exists():
            lh.config(text=hint, fg='#c8a42a')
        self._lbl_status.config(text="Hint shown")

    def _set_level(self, idx):
        self._current_level_idx = idx
        self._selected_category = LEVELS[idx]['category']
        self._update_level_buttons()
        if not self.state.get("waiting_for_ready"):
            old_ids = self.state["player_ids"]
            if self._logger:
                self._logger.save(finished=False)
            self.state = new_state(self._selected_category)
            self.state["player_ids"]         = old_ids
            self.state["waiting_for_ready"]  = False
            self.state["waiting_for_card"]   = True
            self.state["session_start_time"] = time.time()
            self._logger             = SessionLogger(self._selected_category, old_ids)
            self._stable_word        = ''
            self._stable_count       = 0
            self._last_idle_nudge    = 0.0
            self._last_buzzer_reminder = 0.0
            self._last_card_nudge    = 0.0
            self._between_words_sent = False
            self.reachy.send("category", category=self._selected_category)
            lv = LEVELS[idx]
            self._lbl_status.config(
                text=f"Level {idx+1}: {lv['desc']} — show a word card!")
            cs = getattr(self, '_lbl_card_status', None)
            if cs and cs.winfo_exists():
                cs.config(text="Show a word card to the camera!", fg=P['gold'])
            for attr in ('_lbl_feedback', '_lbl_hint'):
                lbl = getattr(self, attr, None)
                if lbl and lbl.winfo_exists():
                    lbl.config(text="")
            self._refresh()

    def _next_category(self):
        self._set_level((self._current_level_idx + 1) % len(LEVELS))

    def _new_game(self):
        if self.state.get("waiting_for_ready"):
            return
        self._kiosk_reset_token += 1   # cancel any pending kiosk reset
        old_ids = self.state["player_ids"]
        if self._logger:
            self._logger.save(finished=False)
        self.state = new_state(self._selected_category)
        self.state["player_ids"]         = old_ids
        self.state["waiting_for_ready"]  = False
        self.state["waiting_for_card"]   = True
        self.state["session_start_time"] = time.time()
        self._logger       = SessionLogger(self._selected_category, old_ids)
        self._start_recording(force=True)   # fresh file for the replay (stops the old)
        self._stable_word  = ''
        self._stable_count = 0
        self._last_idle_nudge    = 0.0
        self._last_buzzer_reminder = 0.0
        self._last_card_nudge    = 0.0
        self._between_words_sent = False
        self.reachy.send("restart")
        for attr in ('_lbl_feedback', '_lbl_hint', '_lbl_invalid'):
            lbl = getattr(self, attr, None)
            if lbl and lbl.winfo_exists():
                lbl.config(text="")
        cs = getattr(self, '_lbl_card_status', None)
        if cs and cs.winfo_exists():
            cs.config(text="Show a word card to the camera to begin!", fg=P['gold'])
        self._lbl_status.config(
            text="New game — show a word card to the camera!")
        self._refresh()


# -----------------------------
# ENTRY POINT
# -----------------------------

def main():
    if not _TK_OK:
        print("ERROR: tkinter is required to run the game GUI.")
        return

    # Enumerate cameras ONCE up front and cache it. pygrabber's DirectShow
    # scan can take a couple of seconds; doing it here means neither the
    # recorder nor the detection camera loop pays that cost again (both reuse
    # the cache), so the detection camera opens promptly.
    try:
        import time as _t
        from camera_select import list_cameras
        _t0 = _t.time()
        _cams = list_cameras()
        print(f"[CAMERA] enumerated {len(_cams)} cameras in "
              f"{_t.time() - _t0:.2f}s: {_cams}")
    except Exception as _e:
        print(f"[CAMERA] camera enumeration failed: {_e}")

    # Recording is now PER GAME (one file per group), owned by the GUI: it
    # starts in _start_game (when players are detected) and stops the instant
    # the session ends. So we do NOT start a recorder here at launch — we build
    # the GUI, which manages its own recorder lifecycle. (The camera list was
    # warmed just above, so the GUI's first _start_recording resolves fast.)
    root = tk.Tk()
    gui = ScrambleGameGUI(root)

    # On Windows, closing the console/terminal window (or a logoff/shutdown)
    # kills the process directly — it does NOT raise KeyboardInterrupt and
    # does NOT run atexit handlers or the `finally` block below in time, so an
    # in-progress recording never finalizes. Catching the console-close event
    # gives us a last chance to finalize the current game's recording before
    # Windows tears the process down.
    if platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            _CTRL_CLOSE_EVENT    = 2
            _CTRL_LOGOFF_EVENT   = 5
            _CTRL_SHUTDOWN_EVENT = 6

            HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def _on_console_ctrl(ctrl_type):
                if ctrl_type in (_CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT,
                                  _CTRL_SHUTDOWN_EVENT):
                    print(f"[REC] Console closing (event {ctrl_type}) — "
                          f"finalizing recording…")
                    gui._stop_recording()
                return False   # let Windows continue its own shutdown handling

            # Keep a reference alive at module scope so it isn't garbage
            # collected — a GC'd ctypes callback would crash on invocation.
            global _console_ctrl_handler_ref
            _console_ctrl_handler_ref = HandlerRoutine(_on_console_ctrl)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(
                _console_ctrl_handler_ref, True)
        except Exception as e:
            print(f"[REC] could not install console-close handler: {e}")

    # Belt-and-suspenders finalize on any normal interpreter exit.
    atexit.register(gui._stop_recording)

    # Let Python process Ctrl-C promptly even while tk's mainloop is blocking.
    def _pump():
        root.after(200, _pump)
    root.after(200, _pump)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[REC] Ctrl-C received — finalizing recording…")
    finally:
        gui._stop_recording()   # idempotent; safe alongside atexit


if __name__ == "__main__":
    main()