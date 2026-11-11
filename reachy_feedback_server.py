"""
Reachy Mini Feedback Server — HARD CODED MODE + GAZE + MOTION
Kid-friendly pacing, face-tracking gaze, expressive gestures.

MERGED BUILD — combines:
  • Full EventRouter feedback for BOTH games (scramble + speed), mode-based
    speech pacing, queue management, built-in dynamic-line pregeneration.
  • Robot-camera player-ID scanning (GET /player_ids), personalized name
    greetings, goodbye/final-score announcements, optional live LLM lines.

RUN ON THE ROBOT:
    source /venvs/apps_venv/bin/activate
    cd ~/Scramble_Words
    python reachy_feedback_server.py

If audio is busy:  sudo fuser -k /dev/snd/*
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import shutil
import subprocess
import threading
import time
import urllib.request as _urlreq
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from typing import Any

try:
    from reachy_mini import ReachyMini
    SDK_OK = True
    SDK_ERR = None
except Exception as exc:
    ReachyMini = None
    SDK_OK = False
    SDK_ERR = exc
    print("[WARN] reachy_mini not available")

try:
    from reachy_mini.utils import create_head_pose
except Exception:
    create_head_pose = None

try:
    import cv2
    CV2_OK = True
    CV2_ERR = None
except Exception as exc:
    cv2 = None
    CV2_OK = False
    CV2_ERR = exc

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


# ── Config ─────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 8765

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPER_MODEL = os.environ.get("PIPER_MODEL", "/home/pollen/en_US-amy-medium.onnx")

AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "reachy_audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

ALSA_DEVICE = "plughw:0,0"

HEAD_Z_MM = 0.0

# ── GAZE CONFIG ──
GAZE_ENABLED       = True     # master switch for face tracking
GAZE_FPS           = 5.0      # frames/sec analysed
GAZE_MAX_YAW_DEG   = 7.0     # how far the head turns left/right at most
GAZE_MAX_PITCH_DEG = 2.0     # how far the head tilts up/down at most
GAZE_MIRROR        = False    # flip if the robot turns the WRONG way (left/right)
GAZE_PITCH_MIRROR  = True     # flip if the robot tilts the WRONG way (up/down)
GAZE_SWITCH_SECONDS = 4.0     # how long to look at each player (2-player mode)
GAZE_SMOOTH        = 0.20     # 0..1, higher = snappier, lower = smoother
GAZE_LOST_FRAMES   = 25       # hold last position this many no-face frames
TABLE_LOOK_PITCH_DEG = 2.0   # how far the head tilts DOWN in "table" mode
FACE_GREET_GAP_S   = 15.0     # face absent this long, then reappears → greet
FACE_CASCADE = (cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if cv2 is not None else None) 
# ── PLAYER-ID SCAN CONFIG ──
# The robot's own camera also scans for ArUco player ID cards (IDs 200-999,
# DICT_6X6_1000 — mirrors PLAYER_ID_RANGE in detection.py on the game machine).
# Detected IDs are served at GET /player_ids so the game can merge them with
# its external-camera detections ("dual-camera login").
ID_SCAN_ENABLED  = True
PLAYER_ID_RANGE  = range(200, 1000)
ID_SCAN_TTL_S    = 1.5          # an ID stays "recent" this long after last sight


def _make_aruco_detect():
    """Return detect(gray) -> (corners, ids) for old & new cv2.aruco APIs."""
    if cv2 is None or not hasattr(cv2, "aruco"):
        return None
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    try:                                    # OpenCV >= 4.7
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dic, params)
        return lambda gray: detector.detectMarkers(gray)[:2]
    except AttributeError:                  # older API
        params = cv2.aruco.DetectorParameters_create()
        return lambda gray: cv2.aruco.detectMarkers(gray, dic,
                                                    parameters=params)[:2]


_ARUCO_DETECT = _make_aruco_detect() if ID_SCAN_ENABLED else None


# ── Load Phrases Dynamically ───────────────────────────

PHRASES_FILE = os.path.join(BASE_DIR, "phrases.json")

try:
    with open(PHRASES_FILE, "r", encoding="utf-8") as f:
        PHRASES = json.load(f)
    print(f"[INFO] Successfully loaded {len(PHRASES)} phrase categories from JSON.")
except Exception as e:
    print(f"[ERROR] Failed to load phrases.json: {e}")
    PHRASES = {}

VALID_PHRASES_SET = {phrase for plist in PHRASES.values() for phrase in plist}


# ── Player names + name greetings ──────────────────────
# Copy player_names.json to the robot alongside this file (same one the game
# machine uses). When a player_detected event arrives with a name, Reachy
# greets the camper personally. Clips are pre-generated at startup so there is
# no TTS lag at camp.

PLAYER_NAMES_FILE = os.path.join(BASE_DIR, "player_names.json")
PLAYER_NAMES = {}


def _load_player_names() -> dict:
    try:
        with open(PLAYER_NAMES_FILE, "r", encoding="utf-8") as f:
            _raw = json.load(f)
        names = {k: (v["name"] if isinstance(v, dict) else str(v))
                 for k, v in _raw.items() if not k.startswith("_")}
        print(f"[INFO] Loaded {len(names)} player names.")
        return names
    except Exception as e:
        print(f"[WARN] player_names.json not loaded ({e}) — generic greetings only.")
        return {}


def reload_player_names() -> tuple[bool, str]:
    global PLAYER_NAMES
    PLAYER_NAMES = _load_player_names()
    if not PLAYER_NAMES:
        return False, "no player names loaded"
    print("[INFO] Refreshing personalized greeting cache...")
    count = 0
    for line in all_greeting_lines():
        if gen(line):
            count += 1
    return True, f"reloaded {len(PLAYER_NAMES)} names, cached {count} greetings"


PLAYER_NAMES = _load_player_names()

GREETING_TEMPLATES = [
    "Hi {name}! Let's play!",
    "Welcome, {name}! Get ready!",
    "There you are, {name}! Game time!",
]

GOODBYE_SCORE_TEMPLATES = {
    # Filled in by the goodbye handler with real names/scores, then spoken.
    # {w1}/{w2} are "word" or "words" (pluralised by the handler).
    "speed_tie":   "Final score: {p1} found {s1} {w1} and {p2} found {s2} {w2}. It's a tie!",
    "speed_win":   "Final score: {p1} found {s1} {w1} and {p2} found {s2} {w2}. {winner} wins!",
    "speed_solo":  "Final score: {p1} found {s1} {w1}. Amazing!",
    "scramble":    "{names} scored {score} out of {total}!",
}


def all_greeting_lines():
    """Every personalised line we might speak — for cache pre-generation.

    IMPORTANT: every line here must match what the game ACTUALLY sends,
    byte-for-byte. The cache is keyed on an md5 of the exact text, so a single
    differing character means a live Piper synth (~1-2s of dead air) every
    time that line is spoken. If you change wording in game.py or
    speed_game.py, change it here too.
    """
    lines = []
    for name in PLAYER_NAMES.values():
        for tpl in GREETING_TEMPLATES:
            lines.append(tpl.format(name=name))
        # game.py's scripted login lines (game.py _announce_new_players) —
        # spoken via say_dynamic, so they MUST be mirrored here byte-for-byte
        # (cache is keyed on exact text, em dash included) or every login
        # pays a live Piper synth. If game.py's wording changes, change it
        # here too.
        lines.append(f"Hi {name}! Now player two — show me your card!")
        lines.append(f"Hi {name}! Everyone's here. Let's play!")

        # speed_game.py's turn-handover lines (speed_game.py _show_handover,
        # sent as say= on "round_start"). These were MISSING, which meant a
        # brand-new camper name paid a live Piper synth on EVERY ONE of the 8
        # turn handovers — the "new lanyard is slow" bug. The first string is
        # a two-part f-string in speed_game.py; it concatenates to exactly
        # this, with a single space between "and" and "press".
        lines.append(f"{name}, you go first. Spell words from the tiles and "
                     f"press the buzzer for each one. Ready? Go!")
        lines.append(f"{name}, your turn! Go!")
    return lines


# ── LIVE LLM FEEDBACK (optional, off by default) ───────────────────────────
# Generates one-off spoken lines DURING play using Ollama running on a Mac on
# the camp LAN (no internet needed — the model is local to the Mac). Only used
# at NATURAL PAUSES (level_complete / goodbye) where a 2-4s thinking gap is
# invisible — never on time-critical buzz results (correct / word_found),
# where the 2.5s budget + live Piper synth of the fresh line caused ~5s of
# dead air after every buzz. Hints, rules, and answer-adjacent speech always
# come from phrases.json / game-built text.
# Every generated line is validated, and ANY failure (timeout, bad output,
# Mac offline) falls back to the normal line — the robot can never go silent
# or off-book.
#
# Enable:   REACHY_LLM=1 OLLAMA_URL=http://10.42.0.1:11434 \
#           python reachy_feedback_server.py
# (Run `OLLAMA_HOST=0.0.0.0 ollama serve` on the Mac so the LAN can reach it.)

LLM_ENABLED    = os.environ.get("REACHY_LLM", "0") == "1"
LLM_BASE_URL   = os.environ.get("OLLAMA_URL", "http://10.42.0.1:11434").rstrip("/")
LLM_MODEL      = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
LLM_TIMEOUT_S  = 2.5    # hard budget — miss it and we use the fallback line
LLM_MAX_WORDS  = 12
LLM_MAX_FAILS  = 3      # consecutive failures before auto-disabling for session

LLM_SYSTEM = (
    "You write ONE short spoken line for Ree chee, a cheerful robot "
    "coaching kids aged 6-10 at a summer camp word game. Rules: maximum "
    f"{LLM_MAX_WORDS} words; natural when spoken aloud; no emojis, hashtags, "
    "or quotation marks; upbeat like a fun camp counselor; never invent or "
    "reveal any answer word other than the one given to you. Output ONLY the "
    "line itself — no explanation, no preamble."
)

_llm_fail_streak = 0


def _llm_validate(text: str) -> str | None:
    """Accept only a clean, short, speakable single line."""
    if not text:
        return None
    line = text.strip().splitlines()[0].strip().strip('"').strip("'").strip()
    if not line or len(line.split()) > LLM_MAX_WORDS:
        return None
    # Printable ASCII only — rejects emojis and formatting artefacts.
    if not all(32 <= ord(ch) < 127 for ch in line):
        return None
    if any(tok in line for tok in ("#", "http", "*", "`", "{", "}")):
        return None
    # Reject preambles the model sometimes emits instead of the line itself.
    low = line.lower()
    if line.endswith(":") or low.startswith(("here is", "here's", "sure",
                                             "okay,", "of course")):
        return None
    return line


def llm_line(user_prompt: str) -> str | None:
    """Ask Ollama for one line. Returns a validated line or None (= fallback).

    Auto-disables after LLM_MAX_FAILS consecutive failures so a Mac that left
    the network doesn't add a timeout to every event for the rest of the day.
    """
    global _llm_fail_streak
    if not LLM_ENABLED or _llm_fail_streak >= LLM_MAX_FAILS:
        return None
    body = json.dumps({
        "model": LLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": 0.9, "num_predict": 30},
    }).encode()
    try:
        req = _urlreq.Request(f"{LLM_BASE_URL}/api/chat", data=body,
                              headers={"Content-Type": "application/json"},
                              method="POST")
        with _urlreq.urlopen(req, timeout=LLM_TIMEOUT_S) as r:
            data = json.loads(r.read().decode())
        line = _llm_validate(data.get("message", {}).get("content", ""))
        if line:
            _llm_fail_streak = 0
            print(f"[LLM] {line}")
            return line
        _llm_fail_streak += 1
        return None
    except Exception as exc:
        _llm_fail_streak += 1
        if _llm_fail_streak == LLM_MAX_FAILS:
            print(f"[LLM] disabled after {LLM_MAX_FAILS} failures ({exc}) — "
                  f"canned lines only from here on.")
        return None


def pick(key: str, last: dict) -> str:
    options = PHRASES.get(key, ["Okay"])
    choices = [p for p in options if p != last.get(key)] or options
    chosen = random.choice(choices)
    last[key] = chosen
    return chosen


# ── TTS (slow-pace for younger ears) ───────────────────
# ONE voice, always. Two things previously made Reachy's voice drift:
#   1. noise_scale was cranked to 1.8 (Piper default 0.667) — that knob adds
#      RANDOM pitch/intonation variation per synthesis, so every clip rolled
#      new prosody dice and sounded like a slightly different speaker.
#   2. The cache was keyed on text alone, so clips generated under OLD
#      settings (0.95 / 1.0, no sox) sat in the cache forever while new
#      lines were made with NEW settings (+ sox pitch shift). Which voice
#      you heard depended on when a line was first cached.
# Fix: all voice knobs live in VOICE below, and they're baked into the cache
# filename — change any value and every line regenerates consistently on
# next use (old files are simply ignored; delete reachy_audio_cache/ to
# reclaim space). Keep noise_scale ≤ ~0.8 or the voice wobbles again.

VOICE = {
    "length_scale": "0.85",   # lower = faster/peppier (1.0 = natural pace)
    "noise_scale":  "0.667",  # KEEP LOW — this is the "different voice" knob
    "noise_w":      "0.80",   # phoneme-length variation (Piper default)
    "sox_pitch":    "150",    # +cents, applied only if sox is installed;
                              # "" disables the pitch lift entirely
}

# Anything that changes the sound is part of the cache identity — including
# whether sox is present, so installing/removing sox can't split the voice.
_VOICE_TAG = "|".join([VOICE["length_scale"], VOICE["noise_scale"],
                       VOICE["noise_w"],
                       VOICE["sox_pitch"] if shutil.which("sox") else "",
                       os.path.basename(PIPER_MODEL)])


def wav_path(text: str) -> str:
    key = hashlib.md5((_VOICE_TAG + "\x00" + text).encode()).hexdigest()
    return os.path.join(AUDIO_CACHE_DIR, key + ".wav")


def gen(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    path = wav_path(text)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    if not os.path.exists(PIPER_MODEL):
        print("[WARN] Missing Piper model")
        return None
    try:
        subprocess.run(
            ["piper", "--model", PIPER_MODEL,
             "--length_scale", VOICE["length_scale"],
             "--noise_scale", VOICE["noise_scale"],
             "--noise_w", VOICE["noise_w"],
             "--output_file", path],
            input=text.encode(), stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, timeout=40, check=True)

        # Optional: nudge pitch up slightly for a brighter, more cheerful
        # tone. Requires `sox` (apt install sox). +100 cents = 1 semitone.
        # Deterministic, so it doesn't cause voice drift — and it's part of
        # _VOICE_TAG, so toggling sox regenerates the cache consistently.
        if VOICE["sox_pitch"] and shutil.which("sox"):
            tmp = path + ".tmp.wav"
            try:
                subprocess.run(
                    ["sox", path, tmp, "pitch", VOICE["sox_pitch"]],
                    check=True, timeout=15,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(tmp, path)
            except Exception as exc:
                print(f"[WARN] pitch shift skipped for {text!r}: {exc}")
                if os.path.exists(tmp):
                    os.remove(tmp)

        return path if os.path.exists(path) else None
    except Exception as exc:
        print(f"[WARN] piper failed {text!r}: {exc}")
        return None


# ── Spoken-feedback transcript ─────────────────────────────
# Every line Reachy actually speaks is appended here — one file per day in the
# logs/ folder next to this script (e.g. logs/reachy_speech_20260722.log), each
# row "YYYY-MM-DD HH:MM:SS<TAB>the spoken line". The rules line ("Here's how we
# play!") and goodbye ("Bye for now!") act as natural per-group delimiters, and
# the timestamps line up with the game logs on the PC. Append-only and fully
# wrapped in try/except so logging can never interrupt or crash speech.
_SPEECH_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _log_spoken(text: str):
    try:
        os.makedirs(_SPEECH_LOG_DIR, exist_ok=True)
        path  = os.path.join(_SPEECH_LOG_DIR,
                             f"reachy_speech_{time.strftime('%Y%m%d')}.log")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{stamp}\t{text}\n")
    except Exception as e:
        print(f"[SPEECH-LOG] WARN: could not write speech log: {e}")


# ── Bot ─────────────────────────────────────────────

class Bot:
    # Per-utterance pacing, by game mode. (cooldown-before, think-pause,
    # breathing-room-after) in seconds. "scramble" is deliberately slow/
    # kid-friendly since that game has long gaps between cards anyway.
    # "speed" needs to keep up with words being submitted every few
    # seconds, so it trims the same ~6.5s floor down to ~2s.
    PACING = {
        "scramble": (3.5, 0.5, 2.5),
        "speed":    (0.8, 0.2, 1.0),
        # Login is a back-and-forth conversation — a kid shows a card and
        # expects "Hi <name>!" right away. The slow scramble pacing here
        # made the greeting land 7-10s after the card was read (prompt
        # speech + 2.5s breathing + 3.5s cooldown + 0.5s think). Near-zero
        # gaps: greeting starts as soon as the previous line's audio ends.
        "login":    (0.3, 0.1, 0.4),
    }

    def __init__(self):
        self._say_q = queue.Queue()
        self._last_phrase = {}
        self.speaking = threading.Event()

        self._last_speak_time = 0
        self.mode = "scramble"
        self.SPEAK_COOLDOWN = self.PACING["scramble"][0]

        self.robot = None
        self.ok = False
        self._stop = False

        # audio routing
        self._spk_lock = threading.Lock()
        self._dev = None
        self._out_sr = 16000
        self._media_audio = False     # True = push through media pipeline (camera on)

        # motion / gesture
        self._gesture = None
        self._gesturing = threading.Event()

        # Called (from the gaze thread) when a face appears after a gap —
        # the router uses it to invite walk-ups to log in.
        self.on_face_appeared = None

        # player-ID scan state (robot camera → GET /player_ids)
        self._scan_lock = threading.Lock()
        self._scan_ids: dict[int, float] = {}   # id -> last-seen time
        self._last_face_time = 0.0              # when gaze last saw a face

        # look mode: "faces" = track faces (instructions, greetings,
        # celebrations); "table" = look DOWN at the tiles while kids work.
        self._look_mode = "faces"
        self._table_blend = 0.0   # 0=faces … 1=fully down; eased, no snapping

        # Standby: after a game ends, the robot goes quiet — antennas droop,
        # no lively idle twitches — but gaze keeps watching. The router
        # clears this when a face appears (wake + login invite) or when any
        # login/game event arrives.
        self.standby = False

        # gaze state
        self._gaze_yaw = 0.0
        self._gaze_pitch = 0.0
        self._gaze_lost = 0
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._focus = None
        self._focus_until = 0.0
        self._snap_side = None

        threading.Thread(target=self._say_worker, daemon=True).start()

    def start(self):
        """Connect, enable motors, wake up, start motion + gaze BEFORE speech."""
        if ReachyMini is None:
            print(f"[WARN] reachy_mini import failed ({SDK_ERR}); voice only.")
            return
        try:
            # Camera needs "default"; when camera is open, aplay fights the
            # media pipeline, so audio is pushed through the pipeline instead.
            backend = "default" if (GAZE_ENABLED and cv2 is not None) else "no_media"
            self.robot = ReachyMini(media_backend=backend).__enter__()
            try:
                self.robot.enable_motors()
                print("[INFO] Motors enabled.")
            except Exception as e:
                print(f"[WARN] enable_motors: {e}")
            try:
                self.robot.wake_up()
                time.sleep(0.8)
                print("[INFO] Reachy awake.")
            except Exception as e:
                print(f"[WARN] wake_up: {e}")
            self.ok = True

            # Animator: antennas + gestures (head is driven by gaze).
            threading.Thread(target=self._animate, daemon=True).start()

            # Gaze: open the media audio path, then start face tracking.
            if GAZE_ENABLED and cv2 is not None:
                try:
                    self.robot.media.start_playing()
                    self._out_sr = self.robot.media.get_output_audio_samplerate()
                    self._media_audio = True
                    print(f"[INFO] Audio via media pipeline @ {self._out_sr} Hz")
                except Exception as e:
                    print(f"[WARN] media audio init failed ({e}); using aplay")
                threading.Thread(target=self._gaze_loop, daemon=True).start()
                print("[INFO] Gaze (face tracking) enabled.")
            elif GAZE_ENABLED:
                print(f"[WARN] gaze wanted but OpenCV missing ({CV2_ERR}); no gaze.")

            print("[INFO] Reachy Mini hardware connected and ready.")
        except Exception as e:
            print(f"[WARN] Could not connect to Reachy hardware: {e}")

    def stop(self):
        self._stop = True
        if self.robot:
            try:
                self.robot.__exit__(None, None, None)
            except Exception:
                pass

    # ---- speech ----

    def say(self, key: str, **kwargs):
        text = pick(key, self._last_phrase)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                print(f"[WARN] Missing format key: {e}")
        self._say_q.put(text)

    def say_raw(self, text: str):
        text = text.strip()
        if text not in VALID_PHRASES_SET:
            print(f"[BLOCKED]: {text}")
            return
        self._say_q.put(text)

    def say_dynamic(self, text: str):
        """Speak free-form text built at runtime (word reveals, clues, hints)
        that can't be pre-enumerated in phrases.json. Unlike say_raw(), this
        is NOT checked against VALID_PHRASES_SET, since clue/word text is
        generated on the fly from game.py's own word list and varies every
        call. gen() already TTS's arbitrary text fine; say_raw()'s allow-list
        only exists to keep ambient/canned lines on-script, not to block
        legitimate dynamic game content.
        """
        text = text.strip()
        if not text:
            return
        self._say_q.put(text)

    def say_trusted(self, text: str):
        """Speak a line the SERVER itself constructed (name greetings, score
        announcements, validated LLM lines). Bypasses the phrases.json
        allow-list — never feed it text straight from a network payload;
        format it into a template or sanitise it here first."""
        text = text.strip()
        if text:
            self._say_q.put(text)

    def busy(self) -> bool:
        """True if the robot is currently speaking or has lines queued up.

        Used to gate ambient nudges (observing/idle/waiting_for_card) so they
        skip outright instead of queueing — an unthrottled FIFO speech queue
        means any burst of events outpaces the ~6s-per-line speak rate and
        backs up for the rest of the session, so by word 15 the robot is
        still narrating word 11. Ambient chatter is the first thing to drop;
        real game-result events use clear_queue() instead (see below) so
        they always reflect the CURRENT state rather than stacking up.
        """
        return self.speaking.is_set() or not self._say_q.empty()

    def clear_queue(self):
        """Drop any backlog of not-yet-spoken lines."""
        try:
            while True:
                self._say_q.get_nowait()
        except queue.Empty:
            pass

    def set_mode(self, mode: str):
        """Switch speech pacing for the given game mode. Unknown modes fall
        back to the slower "scramble" pacing."""
        if mode not in self.PACING:
            mode = "scramble"
        if mode == self.mode:
            return
        self.mode = mode
        self.SPEAK_COOLDOWN = self.PACING[mode][0]
        print(f"[INFO] Speech pacing set to '{mode}' mode "
              f"(cooldown={self.PACING[mode][0]}s)")

    def look(self, mode: str):
        """Where the head points between gestures.
        "faces" — face-tracking gaze, head up at the players (use while
                  greeting, explaining rules, celebrating, announcing).
        "table" — gaze suppressed, head pitched down toward the play area
                  (use while the kids are arranging tiles), so the robot
                  visibly watches the game like an attentive coach.
        """
        if mode in ("faces", "table") and mode != self._look_mode:
            self._look_mode = mode

    def gesture(self, name: str):
        self._gesture = name

    def snap_focus(self, side: str):
        self._snap_side = side

    def _say_worker(self):
        while True:
            text = self._say_q.get()

            _cooldown, think_pause, breathing_room = self.PACING[self.mode]

            now = time.time()
            since = now - self._last_speak_time
            if since < self.SPEAK_COOLDOWN:
                time.sleep(self.SPEAK_COOLDOWN - since)

            # Natural pause so it looks like it's thinking
            time.sleep(think_pause)

            self.speaking.set()
            self._last_speak_time = time.time()

            path = gen(text)
            if path:
                print(f"[SAY] {text}")
                _log_spoken(text)
                self._play(path)

            # Breathing space so lines don't run into each other
            time.sleep(breathing_room)
            self.speaking.clear()

    def _play(self, path):
        # Camera open -> push through media pipeline (aplay conflicts with it).
        if self._media_audio:
            try:
                src_sr, data = wavfile.read(path)
                data = (data.astype(np.float32) / 32768.0) if data.dtype == np.int16 \
                    else data.astype(np.float32)
                if data.ndim > 1:
                    data = data[:, 0]
                if src_sr != self._out_sr:
                    g = gcd(int(src_sr), int(self._out_sr))
                    data = resample_poly(data, self._out_sr // g, src_sr // g)
                samples = np.clip(data, -1.0, 1.0).astype(np.float32).reshape(-1, 1)
                with self._spk_lock:
                    self.robot.media.push_audio_sample(samples)
                time.sleep(len(samples) / self._out_sr + 0.3)
                return
            except Exception as exc:
                print(f"[WARN] media audio failed ({exc}); trying aplay")

        # aplay fallback (voice-only / no camera)
        devices = ([["-D", self._dev]] if self._dev else []) + [
            ["-D", ALSA_DEVICE], ["-D", "plughw:1,0"], ["-D", "default"], []
        ]
        for extra in devices:
            try:
                with self._spk_lock:
                    proc = subprocess.Popen(["aplay", "-q", *extra, path],
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                if proc.wait() == 0:
                    self._dev = extra[1] if extra else None
                    return
            except Exception:
                continue

    # ---- movement ----

    def _head(self, yaw=0, pitch=0, roll=0):
        if create_head_pose is None:
            return None
        try:
            return create_head_pose(z=HEAD_Z_MM, yaw=yaw, pitch=pitch,
                                    roll=roll, mm=True, degrees=True)
        except TypeError:
            try:
                return create_head_pose(z=HEAD_Z_MM, mm=True)
            except Exception:
                return None

    def _move(self, antennas=(33, 33), body_yaw=0, head_yaw=0,
              head_pitch=0, head_roll=0, duration=0.6):
        if not self.ok:
            return
        # Ease between "faces" (blend 0) and "table" (blend 1) so mode
        # switches glide (~0.8s) instead of snapping the head 20 degrees.
        target = 1.0 if self._look_mode == "table" else 0.0
        self._table_blend += (target - self._table_blend) * 0.35
        blend = self._table_blend

        # Gestures keep most of their expressiveness: only 40% of the
        # table tilt applies while a gesture is running, so a nod is still
        # a nod and a celebrate still lifts, even in table mode.
        eff = blend * (0.4 if self._gesturing.is_set() else 1.0)
        head_pitch += eff * TABLE_LOOK_PITCH_DEG

        # Face gaze fades out as the head goes down — no background sway
        # toward faces while pointed at the floor.
        gy = max(-GAZE_MAX_YAW_DEG, min(GAZE_MAX_YAW_DEG, self._gaze_yaw))
        gp = max(-GAZE_MAX_PITCH_DEG, min(GAZE_MAX_PITCH_DEG, self._gaze_pitch))
        head_yaw   += gy * (1.0 - blend)
        head_pitch += gp * (1.0 - blend)
        # If the player is far to a side, let the BODY help reach them —
        # fades out with the gaze as the head goes down.
        BODY_ASSIST_THRESHOLD = 22.0
        if abs(gy) > BODY_ASSIST_THRESHOLD:
            body_yaw += ((gy - math.copysign(BODY_ASSIST_THRESHOLD, gy))
                         * 0.8 * (1.0 - blend))
        kw = {"antennas": [math.radians(antennas[0]), math.radians(antennas[1])],
              "body_yaw": math.radians(body_yaw), "duration": duration,
              "method": "minjerk"}
        h = self._head(head_yaw, head_pitch, head_roll)
        if h is not None:
            kw["head"] = h
        try:
            self.robot.goto_target(**kw)
        except Exception:
            pass

    def _g_nod(self):
        for _ in range(2):
            self._move(antennas=(42, 42), head_pitch=9, duration=0.35)
            self._move(antennas=(34, 34), head_pitch=-2, duration=0.35)

    def _g_celebrate(self):
        for yaw in (16, -16, 12, -12):
            self._move(antennas=(70, 70), body_yaw=yaw, head_pitch=-5, duration=0.28)
            self._move(antennas=(40, 40), body_yaw=-yaw * 0.4, head_pitch=4, duration=0.26)
        self._move(antennas=(50, 50), duration=0.35)

    def _g_shake(self):
        for yaw in (-10, 10, 0):
            self._move(antennas=(28, 28), head_yaw=yaw, duration=0.4)

    def _g_tilt(self):
        self._move(antennas=(40, 24), head_roll=12, duration=0.5)
        time.sleep(0.5)
        self._move(antennas=(33, 33), duration=0.5)

    def _g_point(self):
        self._move(antennas=(60, 60), head_pitch=-6, duration=0.3)
        self._move(antennas=(40, 40), duration=0.3)

    def _animate(self):
        gestures = {"nod": self._g_nod, "celebrate": self._g_celebrate,
                    "shake": self._g_shake, "tilt": self._g_tilt,
                    "point": self._g_point}
        idle = 0.0
        while not self._stop:
            g = self._gesture
            if g:
                self._gesture = None
                self._gesturing.set()
                try:
                    gestures.get(g, self._g_nod)()
                except Exception:
                    pass
                self._gesturing.clear()
                continue
            # Head is gaze-controlled (inside _move). Animate antennas only so
            # speaking looks lively without fighting gaze for head direction.
            if self.speaking.is_set():
                self._move(antennas=(random.uniform(38, 52), random.uniform(38, 52)),
                           duration=random.uniform(0.4, 0.6))
                time.sleep(0.30)
                continue
            now = time.time()
            if now >= idle:
                if self.standby:
                    # Resting pose: antennas drooped, slow settle — visibly
                    # "asleep" but gaze still tracks so it can spot a player.
                    self._move(antennas=(10, 10), duration=1.2)
                    idle = now + random.uniform(3.0, 5.0)
                else:
                    self._move(antennas=(33, 33), duration=0.8)
                    idle = now + random.uniform(1.2, 2.2)
            time.sleep(0.12)

    # ---- GAZE: camera + face tracking ----

    def _gaze_loop(self):
        cascade = cv2.CascadeClassifier(FACE_CASCADE)
        if cascade.empty():
            print("[WARN] face cascade failed to load; gaze off.")
            return
        period = 1.0 / max(1.0, GAZE_FPS)
        self._focus_until = time.time() + GAZE_SWITCH_SECONDS
        while not self._stop:
            t0 = time.time()
            try:
                frame = self.robot.media.get_frame()
            except Exception:
                frame = None
            if frame is not None:
                try:
                    self._update_gaze(frame, cascade)
                except Exception:
                    pass
                try:
                    self._scan_player_ids(frame)
                except Exception:
                    pass
            time.sleep(max(0.0, period - (time.time() - t0)))

    # ---- PLAYER-ID SCAN: ArUco cards on the ROBOT's camera ----

    def _scan_player_ids(self, frame):
        """Detect ArUco player cards (200-999) in a robot-camera frame.

        Runs on the same frames the gaze loop already grabs (~GAZE_FPS), so
        it adds no extra camera load. Sightings are timestamped; the HTTP
        handler serves everything seen within ID_SCAN_TTL_S.
        """
        if _ARUCO_DETECT is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = _ARUCO_DETECT(gray)
        if ids is None or len(ids) == 0:
            return
        now = time.time()
        found = [int(i) for i in np.asarray(ids).flatten()
                 if int(i) in PLAYER_ID_RANGE]
        if not found:
            return
        with self._scan_lock:
            for pid in found:
                if pid not in self._scan_ids:
                    print(f"[ID-SCAN] robot camera sees player card {pid}")
                self._scan_ids[pid] = now

    def get_recent_player_ids(self):
        """Player IDs seen by the robot camera within ID_SCAN_TTL_S."""
        cutoff = time.time() - ID_SCAN_TTL_S
        with self._scan_lock:
            # prune old entries while we're here
            self._scan_ids = {p: t for p, t in self._scan_ids.items()
                              if t >= cutoff}
            return sorted(self._scan_ids.keys())

    def _update_gaze(self, frame, cascade):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        raw = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9,
                                       minSize=(120, 120))
        faces = sorted(((float((x + fw / 2) / w), float((y + fh / 2) / h))
                        for (x, y, fw, fh) in raw), key=lambda p: p[0])

        now = time.time()

        if not faces:
            self._gaze_lost += 1
            if self._gaze_lost >= GAZE_LOST_FRAMES:
                target_cx, target_cy = 0.5, 0.5
                self._focus = None
            else:
                target_cx, target_cy = self._last_cx, self._last_cy
        else:
            self._gaze_lost = 0
            self._last_face_time = now   # presence signal for /player_ids
            # Notify the router (throttled to ~1/s) whenever a face is in
            # view. The router applies the real gates: login phase only,
            # 30s cooldown, not while speaking — so this wakes standby the
            # moment someone appears AND re-invites a lingering kid.
            if (self.on_face_appeared
                    and now - getattr(self, "_last_face_cb", 0) > 1.0):
                self._last_face_cb = now
                try:
                    self.on_face_appeared()
                except Exception:
                    pass
            if len(faces) == 1:
                target_cx, target_cy = faces[0]
                self._focus = "single"
            else:
                if self._snap_side in ("left", "right"):
                    self._focus = self._snap_side
                    self._snap_side = None
                    self._focus_until = now + GAZE_SWITCH_SECONDS
                elif self._focus not in ("left", "right") or now >= self._focus_until:
                    self._focus = "right" if self._focus == "left" else "left"
                    self._focus_until = now + GAZE_SWITCH_SECONDS
                if self._focus == "left":
                    cand = faces[0]
                    cand = cand if cand[0] < 0.55 else (self._last_cx, self._last_cy)
                else:
                    cand = faces[-1]
                    cand = cand if cand[0] > 0.45 else (self._last_cx, self._last_cy)
                target_cx, target_cy = cand
            self._last_cx, self._last_cy = target_cx, target_cy

        # horizontal -> yaw
        offx = (0.5 - target_cx) * 2.0
        if GAZE_MIRROR:
            offx = -offx
        yaw_target = offx * GAZE_MAX_YAW_DEG
        self._gaze_yaw += (yaw_target - self._gaze_yaw) * GAZE_SMOOTH

        # vertical -> pitch
        offy = (0.5 - target_cy) * 2.0
        if GAZE_PITCH_MIRROR:
            offy = -offy
        pitch_target = offy * GAZE_MAX_PITCH_DEG
        self._gaze_pitch += (pitch_target - self._gaze_pitch) * GAZE_SMOOTH


# ── Payload sanitisers ───────────────────────────────────

def _clean_name(value) -> str:
    """Sanitise a name string from a network payload before TTS.

    Allows letters, digits, spaces, and a few joiners; caps the length.
    Returns '' for anything unusable."""
    if not isinstance(value, str):
        return ""
    cleaned = "".join(ch for ch in value.strip()
                      if ch.isalnum() or ch in " -'&,")
    return cleaned[:60]


def _clean_word(value) -> str:
    """Sanitise a game word from a network payload before it reaches an LLM
    prompt or TTS. Letters only, max 12 chars — game words are short."""
    if not isinstance(value, str):
        return ""
    cleaned = "".join(ch for ch in value.strip() if ch.isalpha())
    return cleaned[:12].lower()


# ── Event Router ─────────────────────────────────────────────

class EventRouter:
    # Ambient/nudge events: nice-to-have chatter that should never be allowed
    # to back up. If the robot is already speaking or has something queued,
    # these are dropped outright rather than queued — better to skip a nudge
    # than to have it surface 20 seconds late, narrating a card that's no
    # longer on the table.
    AMBIENT_EVENTS = {
        "observing", "idle", "waiting_for_card", "wrong_letter",
        "word_found", "word_invalid", "press_buzzer",
        # time_warning is ambient ON PURPOSE. It used to be a "real" event,
        # which meant it fell into the clear_queue() branch and always spoke —
        # so a 30-seconds-left announcement beat the kid's own word feedback
        # every time they collided. The kid's word is the thing they're
        # waiting to hear about; the clock is on screen and counting either
        # way. Being ambient means it now gets dropped when the robot is
        # already talking, instead of talking over it.
        "time_warning",
    }

    # Ambient events that respect their cooldown but are NOT dropped when the
    # robot is busy — they queue and play a beat late instead of vanishing.
    # word_found is the one thing in a speed turn that's a direct response to
    # something the kid just DID, so silence is worse than a ~1s delay. Safe
    # to queue because its 4s cooldown caps it at one short line per 4s; the
    # backlog this exists to prevent can't build up.
    NEVER_DROP_AMBIENT = {"word_found"}

    # Per-type cooldown overrides (seconds). Anything not listed uses
    # DEFAULT_AMBIENT_COOLDOWN. word_found/word_invalid fire on nearly every
    # keystroke in speed_game.py's fast-paced round, so they need a much
    # shorter cooldown than the scramble game's once-per-card nudges.
    AMBIENT_COOLDOWNS = {
        "waiting_for_card": 10.0,
        "word_found":       4.0,
        "word_invalid":     4.0,
        # game.py paces idle nudges itself (first at 20s, then every 25s,
        # escalating to real hints) — keep the server gate below that so
        # the escalated hint never gets silently dropped here.
        "idle":             20.0,
        # game.py paces both of these itself (BUZZER_HINT_DELAY_S /
        # BUZZER_HINT_REPEAT_S and WRONG_LETTER_REPEAT_S). These gates are a
        # backstop only and MUST stay just under game.py's numbers — set them
        # higher and the server silently swallows nudges game.py meant to send.
        "press_buzzer":     18.0,
        "wrong_letter":     22.0,
        # Marks are 20s apart (30s, 10s) so this never actually bites — it's
        # here only so time_warning doesn't inherit the 30s default and get
        # gated on top of the busy-drop.
        "time_warning":      3.0,
        "face_greet":       30.0,   # walk-up login invite, at most every 30s
    }
    DEFAULT_AMBIENT_COOLDOWN = 30.0

    # Bookkeeping-only events: never speak, never clear the speech queue.
    # Bookkeeping-only: seated_prompt just tells us the ready screen is up
    # (face_greet does the talking), so it must not clear the speech queue —
    # a goodbye is often still playing when the kiosk resets to that screen.
    SILENT_EVENTS = {"gaze", "session_start", "game_start_silent",
                     "seated_prompt"}

    def __init__(self, bot: Bot):
        self.bot = bot
        self.q = queue.Queue()

        # Where we are in the flow:
        #   "ready"   — pre-login screen; a walk-up is invited to BUZZ
        #   "login"   — scan screen; a walk-up is invited to show a CARD
        #   "playing" — mid-game; faces are the players, no invite
        # Only face_greet reads this to choose its line; everything else
        # treats "ready" and "login" the same.
        self.phase = "login"

        # Text face_greet speaks in the "ready" phase. Supplied by the game
        # via seated_prompt(say=...) so game.py stays the single source of
        # truth for what's on screen; _SEATED_LINE is the fallback if the
        # event never arrived.
        self.ready_say = ""

        # Per-event-type cooldown so one ambient event doesn't suppress a
        # different ambient event (previously a single shared timestamp
        # meant "observing" firing could silently block "waiting_for_card"
        # for 30s, which looked like the robot "forgetting" to prompt).
        self.last_ambient_time: dict[str, float] = {}

    def push(self, e: dict):
        self.q.put(e)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        print("[INFO] Event Router Ready.")
        while True:
            e = self.q.get()
            self._handle(e)

    # ── optional LLM: try a live line, fall back to prepared text ──

    def _say_llm_or_dynamic(self, prompt: str, fallback_text: str):
        """Try a live LLM line; fall back to the given prepared text.

        Runs in a background thread so the 2.5 s LLM budget never blocks the
        event queue. Whichever line wins goes through the normal say queue.
        With REACHY_LLM unset (the default), this is exactly say_dynamic().
        """
        bot = self.bot
        if not LLM_ENABLED:
            bot.say_dynamic(fallback_text)
            return

        def _work():
            line = llm_line(prompt)
            if line:
                bot.say_trusted(line)
            else:
                bot.say_dynamic(fallback_text)
        threading.Thread(target=_work, daemon=True).start()

    def _handle(self, e: dict):
        bot = self.bot
        t = e.get("type", "")
        now = time.time()

        # Track the game phase for the standby/walk-up invite AND speech
        # pacing: login lines are rapid back-and-forth; gameplay lines keep
        # the slower kid-friendly cadence.
        # NOTE: game_start only RECORDS the mode — it fires at app launch
        # while the scan screen is still up, so flipping phase to "playing"
        # here would disable the walk-up invite on the very first screen.
        if t == "game_start":
            self.game_mode = e.get("mode", "scramble")
        if t in ("scan_id_prompt", "login_prompt", "robot_intro", "goodbye",
                 "gameover", "player_detected", "login_detected"):
            self.phase = "login"
            bot.set_mode("login")
            if t in ("scan_id_prompt", "login_prompt", "player_detected", "login_detected"):
                bot.standby = False   # someone is actively logging in
                bot.clear_queue()
        elif t == "seated_prompt":
            # The pre-login "sit down, then buzz" screen is up. Still carries
            # the buzzer line via `say`, and must NOT clear the queue (the
            # kiosk reset sends goodbye just before this — clearing would cut
            # it off mid-word).
            #
            # RELIABILITY FIX: the spoken buzzer invite used to come ONLY from
            # face_greet's ~1/s tick (then a 30s cooldown). A quick buzzer
            # press — or the same testers staying seated — beats that tick, so
            # a group can get no prompt at all. So if someone is already in
            # front of the robot when this screen appears, speak the invite
            # right away, QUEUED after any goodbye still playing. Empty room =>
            # no recent face => stays silent, and face_greet still catches a
            # later walk-up.
            self.phase    = "ready"
            self.ready_say = e.get("say", "") or self.ready_say
            bot.set_mode("login")
            face_recent = (now - getattr(bot, "_last_face_time", 0.0)) < 3.0
            if face_recent:
                bot.standby = False                          # someone's here
                self.last_ambient_time["face_greet"] = now   # don't double-speak
                bot.say_dynamic(self.ready_say or _SEATED_LINE)
        elif t in ("card_detected", "card_changed",
                   "press_buzzer", "round_start", "session_start"):
            self.phase = "playing"
            bot.standby = False
            bot.set_mode(getattr(self, "game_mode", "scramble"))
        elif t in ("ready_prompt", "player_welcome"):
            # The rules are being spoken — everyone is already logged in.
            #
            # PHASE and PACING are different things and this branch used to
            # conflate them. It set phase = "login", which is what face_greet
            # reads to pick its line, so after the rules finished the next
            # face to appear got "Player one! Show me your name card to play!"
            # at players who had just logged in and were waiting for their
            # first card. Phase is "playing" from the rules onward.
            #
            # The PACING half of the original intent still stands, and is done
            # by set_mode() alone: ready_prompt arrives ~1.5s after the second
            # login greeting, while that greeting is often still queued.
            # Flipping to scramble pacing here made the queued greeting AND
            # the rules line each pay the slow cadence (3.5s cooldown + 0.5
            # think + 2.5 breathing) — a big chunk of the ~15s post-login
            # silence. So: login PACING through the rules, but not login
            # PHASE. card_detected (the first real gameplay event, seconds
            # later) flips pacing to scramble.
            self.phase = "playing"
            bot.set_mode("login")   # pacing only — NOT a phase declaration
            bot.standby = False

        # A face just appeared after a gap (from the gaze thread). During
        # the login screen, invite them to scan in; during play, ignore —
        # it's just the players moving around.
        if t == "face_greet":
            if self.phase not in ("login", "ready"):
                return
            # Waking from standby is instant and free (posture perks up even
            # if the spoken invite is still on cooldown).
            was_standby = bot.standby
            bot.standby = False
            cooldown = self.AMBIENT_COOLDOWNS.get("face_greet", 30.0)
            if (now - self.last_ambient_time.get("face_greet", 0)) < cooldown:
                return
            if bot.busy():
                return
            self.last_ambient_time["face_greet"] = now
            bot.look("faces")
            bot.gesture("celebrate" if was_standby else "point")
            # WHICH invite depends on where the game is. On the pre-login
            # ready screen the only thing we want from them is a buzzer
            # press — asking for a name card there is premature and
            # contradicts what's on screen. Once the scan screen is up
            # (phase flips back to "login"), the card invite is right.
            # Both lines are pre-cached; see _all_lines().
            if self.phase == "ready":
                bot.say_dynamic(self.ready_say or _SEATED_LINE)
            else:
                bot.say_dynamic("Player one! Show me your name card to play!")
            return

        if t in self.AMBIENT_EVENTS:
            # Per-type cooldown (separate from the busy-check below — this
            # stops the SAME nudge firing every tick once it's allowed to).
            cooldown = self.AMBIENT_COOLDOWNS.get(t, self.DEFAULT_AMBIENT_COOLDOWN)
            if (now - self.last_ambient_time.get(t, 0)) < cooldown:
                return
            # Drop ambient chatter entirely if the robot is mid-speech or
            # already has lines queued — this is what stops the backlog
            # that made the robot sound like it was narrating an old card.
            # NEVER_DROP_AMBIENT events skip this and queue instead: better a
            # late "Nice!" than none. This is the whole priority order —
            # word_found queues through, time_warning gets dropped.
            if bot.busy() and t not in self.NEVER_DROP_AMBIENT:
                return
            self.last_ambient_time[t] = now
        elif t in self.SILENT_EVENTS or t == "game_start":
            # Bookkeeping-only events that never speak and must NEVER touch
            # the speech queue. "gaze" fires every ~0.7s during active play
            # for face-tracking — if it fell into the clear_queue() branch
            # below (it's not "ambient" and has no speech of its own), every
            # single gaze update would wipe out real game speech almost as
            # soon as it was queued, silencing the robot through most of
            # actual gameplay. "session_start"/"game_start" are similar:
            # they carry no rules/speech text of their own and are sent on
            # independent threads alongside a real speech event
            # ("ready_prompt"/"robot_intro") — letting them clear_queue()
            # created a race where the bookkeeping event could land second
            # and wipe out the real line right before it was spoken.
            pass
        else:
            # Real game-state event (card detected, right/wrong, reveal,
            # rules, goodbye, etc.) — always clear any stale backlog first so
            # the robot reacts to what's happening NOW, not whatever ambient
            # chatter piled up while it was busy earlier.
            bot.clear_queue()

        # ── EXPLICIT RULES & ROUTING FLOWS ──

        if t == "robot_intro":
            bot.look("faces")
            bot.gesture("celebrate")
            bot.say("intro")

        # game.py fires this ~500ms after launch just to tell the robot which
        # game is running. The intro line was already spoken once at server
        # boot (see main()) — don't say it again, just acknowledge with a
        # gesture so it doesn't sound like the robot is repeating itself.
        # Also switches speech pacing: scramble keeps the slow kid-friendly
        # cadence, speed mode trims the same pacing down so feedback can
        # keep up with words being submitted every few seconds.
        elif t == "game_start":
            bot.set_mode(e.get("mode", "scramble"))
            bot.gesture("nod")

        # Game just launched → ask the player to scan/log in with their ID
        # card. game.py sends "scan_id_prompt"; "login_prompt" kept as an
        # alias for older builds.
        elif t in ("scan_id_prompt", "login_prompt"):
            bot.look("faces")
            bot.gesture("point")
            say_text = e.get("say", "")
            if say_text:
                bot.say_dynamic(say_text)   # "Player one! Show me your..."
            else:
                bot.say("login_prompt")

        # A player's ID card was successfully read during the scan screen.
        # If the payload carries a camper name, greet them personally;
        # otherwise fall back to the canned line.
        elif t == "player_detected":
            bot.look("faces")
            bot.gesture("celebrate")
            _say_parts(bot, e, "player_detected")

        elif t == "login_detected":
            bot.look("faces")
            bot.gesture("point")
            say_text = e.get("say", "")
            if say_text:
                bot.say_dynamic(say_text)
            else:
                bot.say("login_detected")

        # After the FIRST player logs in, game.py asks (once) whether a
        # second friend wants to scan in too.
        elif t == "second_player_prompt":
            bot.look("faces")
            bot.gesture("tilt")
            say_text = e.get("say", "")
            if say_text:
                bot.say_dynamic(say_text)

        # game.py fires "session_start" right before "ready_prompt", purely
        # as a bookkeeping signal (session timer starts, logger opens) — it
        # carries no rules text of its own. It used to also trigger speech
        # here, sharing a branch with "ready_prompt". Since the two are sent
        # as separate HTTP requests on separate threads, they could arrive
        # out of order; when session_start landed AFTER ready_prompt, its
        # clear_queue() would wipe the real rules text right before it was
        # spoken, leaving only a generic fallback (or nothing). Splitting
        # them so session_start does nothing here removes that race
        # entirely — "ready_prompt" (with the actual rules text attached
        # via `say`) is now the only thing that speaks.
        elif t == "session_start":
            pass

        # Game start confirmed → explain the rules, then play begins.
        # game.py includes the exact rules text via `say` (single source of
        # truth — what's on screen is what gets spoken); only fall back to
        # the canned phrases.json line if it's missing.
        elif t in ("ready_prompt", "player_welcome", "explain_rules"):
            bot.look("faces")   # instructions are for the PLAYERS — look up
            bot.gesture("nod")
            _say_parts(bot, e, "explain_rules")

        # A word card was scanned. Acknowledge that a puzzle is ready WITHOUT
        # saying the word itself — that would hand the kid the answer before
        # they've touched a tile. The word only comes out later, either as
        # praise on `correct` or as the reveal on `word_failed`.
        elif t == "card_detected":
            bot.look("table")   # puzzle started — watch the tiles
            bot.gesture("point")
            # Name the exact tiles to fetch, but in ALPHABETICAL order so the
            # spelling isn't given away — the kid still has to unscramble.
            # e.g. word "BARN" -> "A, B, N, R". Repeated letters are kept
            # (e.g. "LLAMA" -> "A, A, L, L, M") so the tile count is right.
            word = (e.get("word") or "").strip()
            if word:
                letters = ", ".join(sorted(word.upper()))
                bot.say_dynamic(
                    f"Put the letters {letters} in the playing area "
                    f"and arrange them."
                )
            else:
                # No word came through — fall back to the generic nudge.
                bot.say("card_detected")

        # Player swapped in a different card mid-round, abandoning the word
        # that was in progress. Same "don't say the word" rule as
        # card_detected — just acknowledge the switch.
        elif t == "card_changed":
            bot.look("table")
            bot.gesture("point")
            bot.say_dynamic("New card! Let's start that one instead.")

        # Same card shown again — already played this round.
        elif t == "card_repeated":
            bot.gesture("shake")
            word = e.get("word", "")
            bot.say_dynamic(f"Oops! We did {word.upper()} already. Pick a new card!")

        # No card on the table yet — nudge the player to show one.
        elif t == "waiting_for_card":
            bot.gesture("tilt")
            bot.say("waiting_for_card")

        # A letter outside the target word was placed on the table.
        elif t == "wrong_letter":
            # Speaks, but rarely — game.py gates this to once per
            # WRONG_LETTER_REPEAT_S (see the cooldown note below). It fires
            # while the kid is still ARRANGING, so at the old cadence it
            # delivered a verdict on unsubmitted work and then the real
            # verdict contradicted it seconds later. Slowed down it reads as
            # the robot noticing something, not judging.
            bot.gesture("shake")
            bot.say("wrong_letter")

        # Wrong attempt that still has tries left. game.py sends `attempt`
        # (the attempt number just used), not `consecutive_fails` — reading
        # the wrong key meant this always fell through to "wrong_attempt1".
        elif t == "wrong":
            attempt = e.get("attempt", 1)
            bot.gesture("shake")
            # game.py attaches the tailored hint text in `say` — speak it as
            # ONE crisp line: short canned encouragement + actual help,
            # instead of a long platitude that leaves the kid no wiser.
            key = "wrong_attempt2" if attempt >= 2 else "wrong_attempt1"
            prefix = pick(key, bot._last_phrase)
            say_text = e.get("say", "").strip()
            bot.say_dynamic(f"{prefix} {say_text}" if say_text else prefix)

        # Out of attempts — reveal the word AND its clue/description. This is
        # the "robot gives description of the word" moment. game.py's `say`
        # field reveals the word but doesn't include the clue, so build the
        # spoken line here from `target` + `clue` directly instead.
        elif t == "word_failed":
            bot.look("faces")   # the reveal is spoken to the kid
            bot.gesture("tilt")
            target = e.get("target", "")
            clue = e.get("clue", "")
            if target:
                text = f"Ooh, tricky! It was {target.upper()}."
                if clue:
                    text += f" {clue}"
                text += " No worries. New card!"
                bot.say_dynamic(text)
            else:
                bot.say("word_failed")

        # Correct answer — celebrate and read back the clue if we have one.
        # NO LLM here (moved to level_complete/goodbye): a buzz needs an
        # INSTANT reaction, and the LLM cost was 2.5s Ollama budget + live
        # Piper synth of the fresh line (~2s more) — dead air at the game's
        # most time-critical moment. The prepared text below is built from
        # the word lists, so it's in the pregenerated cache and plays fast.
        elif t == "correct":
            bot.look("faces")   # celebrate WITH the kid — look up at them
            bot.gesture("celebrate")
            target = e.get("target", "")
            clue = e.get("clue", "")
            streak = int(e.get("streak", 0) or 0)
            if target:
                text = f"Woo hoo! {target.upper()}! You got it!"
                if streak >= 2:
                    text += f" That's {streak} in a row!"
                if clue:
                    text += f" {clue}"
                # Explicit next step — don't leave the kid guessing.
                text += " Bring a new card to the playing area!"
                bot.say_dynamic(text)
            else:
                bot.say("correct")
            self.last_ambient_time["observing"] = now

        # Explicit hint request (H key) — speak the tailored hint text built
        # by game.py's build_hint().
        elif t == "hint":
            bot.gesture("point")
            say_text = e.get("say", "")
            if say_text:
                bot.say_dynamic(say_text)

        # Tiles already spell the target word correctly — the kid just
        # hasn't pressed the buzzer yet. Distinct from "idle": there's no
        # hint to give here, so this never escalates and never mentions
        # swapping/rearranging tiles — the arrangement is already right.
        elif t == "press_buzzer":
            # The collision-prone one. game.py used to fire this the instant
            # the tiles spelled the target — the same instant the kid reaches
            # for the buzzer — so the line was still playing when the press
            # landed, and clear_queue() cannot stop audio that has already
            # started. game.py now holds it for BUZZER_HINT_DELAY_S first, so
            # anyone who presses promptly never triggers it at all; it only
            # reaches a kid who has genuinely stalled, and a stalled kid is by
            # definition not about to collide with it.
            bot.gesture("point")
            bot.say_dynamic("Looks right! Press the buzzer!")

        # Player has been stuck on the same word for a while. First nudge is
        # a gentle canned line; when game.py escalates it attaches a real
        # hint in `say` — speak that so the robot actually helps.
        elif t == "idle":
            bot.gesture("tilt")
            say_text = e.get("say", "").strip()
            if say_text:
                bot.say_dynamic(say_text)
            else:
                bot.say("idle")

        # NATURAL-PAUSE LLM HOOK: level_complete is a breather — no kid is
        # waiting on a buzz result, and a couple of seconds of "thinking"
        # reads as personality, not lag. Try a live personalized line
        # (validated, 2.5s budget); any failure speaks the cached
        # phrases.json line.
        elif t == "level_complete":
            bot.gesture("celebrate")
            score = int(e.get("score", 0) or 0)
            total = int(e.get("total", 0) or 0)
            fallback = pick("level_complete", bot._last_phrase)
            ctx = (f"The kids just finished a level of the word scramble, "
                   f"solving {score} of {total} words. Write one short "
                   "excited celebration of finishing the level.")
            self._say_llm_or_dynamic(ctx, fallback)

        elif t == "level_up":
            bot.gesture("nod")
            bot.say("level_up")

        elif t == "session_timeout":
            bot.gesture("tilt")
            bot.say("session_timeout")

        elif t == "restart":
            bot.gesture("nod")
            bot.say("restart")

        # End of game → announce the final score, then a warm farewell.
        elif t == "goodbye":
            bot.look("faces")
            self._handle_goodbye(e)

        elif t == "observing":
            # Silent — watch the tiles, look interested (gesture only).
            bot.look("table")
            bot.gesture("tilt")

        # ── speed_game.py events ────────────────────────────────────────
        # (speed_game.py was already sending these — the router just had no
        # handlers for any of them, so the robot said nothing all game.)

        # Start of a player's 2-minute turn. speed_game.py sends the full
        # rules in `say` for Player 1's turn (single source of truth, same
        # pattern as the scramble game's ready_prompt) and a short "your
        # turn" prompt for Player 2 — so the rules are only explained once,
        # right after sign-in, specific to this game mode.
        elif t == "round_start":
            bot.look("faces")   # rules/turn prompt → then watch the table
            bot.gesture("nod")
            say_text = e.get("say", "")
            if say_text:
                bot.say_dynamic(say_text)
            else:
                player = e.get("player", "Player")
                bot.say_dynamic(f"{player}, your two minutes start now. Go!")

        # End of a player's turn — announce their score.
        elif t == "round_end":
            bot.gesture("celebrate")
            player = e.get("player", "?")
            score = e.get("score", 0)
            word = "word" if score == 1 else "words"
            bot.say_dynamic(f"Time's up, Player {player}! You found {score} {word}.")

        # A submitted word was accepted. Ambient — only speaks if the robot
        # is free, since players can submit several words in quick
        # succession; most submissions just get GUI feedback.
        elif t == "word_found":
            bot.look("table")
            bot.gesture("nod")
            # The cheer does NOT name the word. It used to say
            # "Nice, WORD!", which was only pre-cached for the legacy
            # per-player pools (~112 words). speed_game.py now plays from a
            # SHARED pool (ETAOINSHRDLUCP) whose dictionary is ~1.5k words,
            # so almost every submission missed the cache and paid a live
            # Piper synth + sox pass on the robot's CPU — the cheer landed
            # ~10s after the buzzer. These fixed lines are always cached, so
            # feedback costs pacing only (~1-2s), and nothing here has to be
            # regenerated when the pool or word_list.txt changes. The word
            # itself is already on the game screen in green; the robot
            # repeating it added latency for no information.
            # NO LLM here either (moved to level_complete/goodbye): a 2.5s
            # LLM budget meant the cheer arrived about the PREVIOUS word.
            bot.say_dynamic(random.choice(_WORD_FOUND_LINES))
            # FIRST word of the turn only: follow the cheer with a nudge that
            # spells out the loop, because a kid who just learned "press the
            # buzzer" often thinks one word ends their turn. `total` is this
            # player's running count (from speed_game._submit_word); == 1 means
            # this is their first. It queues AFTER the cheer, so they still get
            # the "Nice!" confirmation first, then "keep going". Cached, so no
            # synth cost; only fires once per turn so it can't nag.
            if e.get("total") == 1:
                bot.say_dynamic(_KEEP_GOING_LINE)

        # A submitted word was rejected. Also ambient for the same reason as
        # word_found — short, skippable feedback, not a critical reveal.
        elif t == "word_invalid":
            bot.gesture("shake")
            reason = e.get("reason", "")
            lines = {
                "too_short":  "Needs more letters!",
                "duplicate":  "Already found that one!",
                "not_a_word": "That's not a word!",
                "bad_letters": "Those letters aren't in your pool!",
            }
            bot.say_dynamic(lines.get(reason, "Try again!"))

        # 30 seconds left in the current turn — time-critical, always speak.
        # Time-remaining check-in — time-critical, always speak.
        # speed_game.py sends this at every 30-second boundary
        # (3:30, 3:00, 2:30 … 0:30 for the 4-minute round) with the exact
        # spoken phrase in `say` (built by its _time_left_phrase()). Speak
        # that; fall back to the classic 30-second line only if it's missing
        # (e.g. an older game build that only fired once at 30s).
        elif t == "time_warning":
            bot.gesture("tilt")
            say_text = e.get("say", "").strip()
            bot.say_dynamic(say_text if say_text else "Thirty seconds left!")

        # Turn timer hit zero — time-critical, always speak.
        elif t == "time_up":
            bot.gesture("celebrate")
            player = e.get("player", "?")
            bot.say_dynamic(f"Time's up, Player {player}!")

        # Both turns complete — announce the winner.
        elif t == "gameover":
            bot.look("faces")
            bot.gesture("celebrate")
            p1 = e.get("p1_score", 0)
            p2 = e.get("p2_score", 0)
            winner = e.get("winner", 0)
            if winner == 1:
                text = f"Player one wins, {p1} to {p2}! Great game!"
            elif winner == 2:
                text = f"Player two wins, {p2} to {p1}! Great game!"
            else:
                text = f"It's a tie, {p1} to {p2}! Great game, both of you!"
            bot.say_dynamic(text)

    # ── Goodbye: score announcement + farewell ──────────────────────────────

    def _handle_goodbye(self, e: dict):
        """Announce the final score and say a warm farewell.

        Payloads (from the games):
          speed_game: {mode:"speed", p1_name, p1_score, p2_name, p2_score}
          game.py:    {mode:"scramble", names, score, total}
        Optional: reason ∈ {finished, timeout, inactivity}.
        All payload text is sanitised before being formatted into templates.
        """
        bot = self.bot
        mode = e.get("mode", "")

        try:
            if mode == "speed":
                p1 = _clean_name(e.get("p1_name")) or "Player one"
                p2 = _clean_name(e.get("p2_name"))
                s1 = int(e.get("p1_score", 0))
                s2 = int(e.get("p2_score", 0))
                w1 = "word" if s1 == 1 else "words"
                w2 = "word" if s2 == 1 else "words"
                if not p2:                          # only one round was played
                    line = GOODBYE_SCORE_TEMPLATES["speed_solo"].format(
                        p1=p1, s1=s1, w1=w1)
                elif s1 == s2:
                    line = GOODBYE_SCORE_TEMPLATES["speed_tie"].format(
                        p1=p1, s1=s1, w1=w1, p2=p2, s2=s2, w2=w2)
                else:
                    line = GOODBYE_SCORE_TEMPLATES["speed_win"].format(
                        p1=p1, s1=s1, w1=w1, p2=p2, s2=s2, w2=w2,
                        winner=p1 if s1 > s2 else p2)
            elif mode == "scramble":
                names = _clean_name(e.get("names")) or "You"
                line = GOODBYE_SCORE_TEMPLATES["scramble"].format(
                    names=names,
                    score=int(e.get("score", 0)),
                    total=int(e.get("total", 0)))
            else:
                line = None
        except Exception as exc:
            print(f"[WARN] goodbye payload malformed: {exc}")
            line = None

        bot.gesture("celebrate")
        if line:
            bot.say_trusted(line)
        # NATURAL-PAUSE LLM HOOK: the score line above is speaking (~4s of
        # audio) while the LLM thinks, so its 2.5s budget is completely
        # hidden — the farewell lands right after the score either way.
        # Any failure speaks the cached phrases.json goodbye.
        farewell = pick("goodbye", bot._last_phrase)
        names = _clean_name(e.get("names")) or ""
        ctx = ("The word game session just ended"
               + (f" for {names}" if names else "")
               + ". Write one short warm goodbye telling the kids they "
                 "played well and you hope to see them again.")
        self._say_llm_or_dynamic(ctx, farewell)
        # After the farewell: standby. Antennas droop, no chatter — the
        # robot quietly watches until face_greet wakes it for the next
        # player. (Speech already queued above still plays out normally.)
        bot.standby = True


# ── Server Initialization ───────────────────────────────

bot = Bot()
router = EventRouter(bot)
# Gaze thread → router: a face appearing after a gap becomes a face_greet
# event (spoken login invite while on the scan screen).
bot.on_face_appeared = lambda: router.push({"type": "face_greet"})


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/event":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            router.push(data)
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        if self.path == "/player_ids":
            face_ago = (round(time.time() - bot._last_face_time, 1)
                        if bot._last_face_time > 0 else None)
            body = json.dumps({
                "ids":           bot.get_recent_player_ids(),
                "face_seen_ago": face_ago,   # secs since gaze last saw a face
                "ts":            time.time(),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/busy":
            body = json.dumps({"busy": bot.busy(), "ts": time.time()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/reload_player_names":
            ok, msg = reload_player_names()
            body = json.dumps({"reloaded": ok, "message": msg}).encode()
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/speech_log"):
            # Serve a spoken-feedback transcript so the game machine can mirror
            # it automatically (no manual scp). Defaults to today's file;
            # optional ?date=YYYYMMDD fetches a specific day.
            from urllib.parse import urlparse, parse_qs
            q   = parse_qs(urlparse(self.path).query)
            day = (q.get("date", [None])[0]) or time.strftime("%Y%m%d")
            fpath = os.path.join(_SPEECH_LOG_DIR, f"reachy_speech_{day}.log")
            try:
                with open(fpath, "rb") as f:
                    body = f.read()
            except OSError:
                body = b""   # no transcript yet — return empty, not an error
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *a):
        return


DYNAMIC_LINES_FILE = os.path.join(BASE_DIR, "dynamic_lines.json")

# ── Built-in dynamic lines ──────────────────────────────────────────────
# Mirrors build_speech_manifest.py's logic, but with the word data baked in
# directly so pregeneration works without running anything on the laptop or
# copying a dynamic_lines.json over. IMPORTANT: if CATEGORIES["animals"],
# WORD_CLUES, or the speed-game word pools change in game.py/speed_game.py,
# update the copies below too, or newly-edited words won't be pre-cached
# (they'll still work fine — just with a one-time Piper delay on first use).

_ANIMAL_WORDS = [
    "ant", "bat", "cat", "cod", "cow", "doe", "dog", "hen", "hog",
    "owl", "ram", "rat",
    "bear", "boar", "clam", "colt", "crab", "crow", "gnat", "goat",
    "hare", "lamb", "mare", "mole", "moth", "newt", "toad", "worm",
    "wren",
]

_ANIMAL_CLUES = {
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

_SCRAMBLE_RULES = (
    "Here is how to play. Pick a card and show it to my camera. "
    "Unscramble the letters, then press the buzzer. "
    "Show me your first card!"
)

_SPEED_P1_WORDS = {
    'hip', 'his', 'hit', 'hop', 'hot', 'ish', 'lip', 'lit', 'lop', 'lot',
    'ohs', 'oil', 'opt', 'pit', 'pot', 'sip', 'sit', 'sop', 'sot', 'til',
    'tip', 'top', 'tho',
    'hilt', 'hips', 'hits', 'hops', 'host', 'hots', 'lips', 'lisp', 'list',
    'lops', 'lost', 'lots', 'oils', 'opts', 'pith', 'pits', 'plot', 'posh',
    'post', 'pots', 'ship', 'shop', 'shot', 'silo', 'silt', 'slip', 'slit',
    'slop', 'slot', 'soil', 'spit', 'spot', 'stop', 'this', 'tips', 'toil',
    'tops', 'tosh',
    'hilts', 'hoist', 'pilot', 'plots', 'polis', 'posit', 'spilt', 'split',
    'spoil', 'toils',
    'pilots', 'polish', 'spoilt',
}
_SPEED_P2_WORDS = {
    'ace', 'and', 'ane', 'arc', 'are', 'can', 'car', 'cud', 'cue', 'cur',
    'dan', 'dar', 'den', 'due', 'dun', 'ear', 'end', 'era', 'ran', 'red',
    'rue', 'run',
    'aced', 'acne', 'acre', 'cane', 'card', 'care', 'cran', 'crud', 'curd',
    'cure', 'dace', 'dance', 'dare', 'darn', 'dean', 'dear', 'duce', 'dune',
    'dunce', 'dure', 'earn', 'near', 'nerd', 'nude', 'race', 'rand', 'read',
    'rend', 'rude', 'rune', 'urea',
    'cadre', 'caned', 'caner', 'cared', 'cedar', 'crane', 'crude', 'cured',
    'dance', 'dunce', 'raced', 'rance', 'under',
    'dancer', 'craned', 'durance',
}

_SPEED_RULES_P1 = (
    "You each have seven letter tiles. Build as many words as "
    "you can before time runs out — every word needs at least "
    "three letters. Press the buzzer to submit a word. "
    "You have two minutes, Player one. Go!"
)
_SPEED_RULES_P2 = "Your turn, Player two! Two minutes on the clock. Go!"

# Spoken on every accepted word in speed mode. Deliberately word-agnostic —
# see the word_found handler for why. Read by BOTH the handler and the
# pregeneration list below, so there is no byte-for-byte mirror to drift out
# of sync: whatever is in this tuple is what gets cached. Add lines freely
# (keep them 1-2 syllables, ages 6-12); each new one costs one 1-2s synth on
# the next --pregenerate run and is free forever after.
def _say_parts(bot, e, fallback_event):
    """Speak an event's text, preferring `lines` (a list) over `say` (one string).

    Why a list: greetings are the one place where names have to be spoken, and
    a name PAIR ("Hi Alice and Bob!") is combinatorially uncacheable — N^2
    entries, so every new camper pairing pays a live Piper synth. Sent as
    separate components ["Hi Alice!", "Hi Bob!", "Everyone's here. Let's play!"]
    each part is individually cacheable: N per-name clips plus a fixed tail.
    They queue back-to-back and play as one continuous greeting.

    This also fixes the greeting getting eaten: game.py used to send one
    player_detected PER player, and player_detected calls clear_queue() in the
    phase-routing block above — so the second player's event wiped the first
    player's hello before it was ever spoken. One event, many lines, one clear.
    """
    parts = e.get("lines") or []
    if isinstance(parts, str):
        parts = [parts]
    if not parts:
        say_text = e.get("say", "")
        parts = [say_text] if say_text else []
    if parts:
        for part in parts:
            if part:
                bot.say_dynamic(part)
    else:
        bot.say(fallback_event)


# Spoken once, right after a player's FIRST accepted word in a turn — see the
# word_found handler. Teaches the core loop: a word doesn't end the turn, keep
# making more until the clock runs out. Single fixed string (not a random set)
# so it's unmistakable the one time it matters, and trivially cacheable.
_KEEP_GOING_LINE = "Great start! Keep going — make another word and buzz again!"

_WORD_FOUND_LINES = (
    # Short beats — the workhorses. Fire on most words.
    "Nice!",
    "Got one!",
    "Yes!",
    "Good one!",
    "Nice work!",
    "Awesome!",
    "Sweet!",
    "Way to go!",
    # Bigger celebrations. Note "Woo hoo" is TWO words: that spelling is what
    # the scramble game's correct line already uses, and Piper says it right.
    # "Woohoo" as one word comes out mangled.
    "Woo hoo! Nailed it!",
    "That's it! Nice one!",
    "You did it!",
    "Boom! Nice one!",
)

# Pre-login "seated and ready?" invite. Byte-for-byte identical to
# game.py's _SEATED_SAY and speed_game.py's ready_prompt say= string — the
# audio cache is keyed on an md5 of the exact text, so all three sharing one
# string means one cached clip serves both games. Used as face_greet's
# fallback and pre-cached in _all_lines().
_SEATED_LINE = ("Hello! When you're seated and ready, press the buzzer. "
                "Then we'll get started.")

# speed_game.py's welcome rules block, sent as ONE component of the welcome
# (see speed_game.py _show_welcome_screen). Name-free on purpose: that's what
# makes it a single cached clip instead of one per name pairing. Must stay
# byte-for-byte identical to the copy in speed_game.py.
_SPEED_WELCOME_RULES = (
    "This is Speed Builder: each player gets one minute per turn, "
    "and there are eight turns total. "
    "Make as many words as you can from the shared 14-letter pool.")


def _build_hint(target: str, level: int, detected: str = "") -> str:
    """Mirrors game.py's build_hint() exactly — needed so the cached text
    matches what the server actually requests via say_dynamic()."""
    t = target.upper()
    d = (detected or "").upper()
    if level <= 1:
        clue = _ANIMAL_CLUES.get(target.lower())
        if clue:
            return f"Here is a clue. {clue}"
        if d and d[0] == t[0]:
            return f"Good start! The word ends with {t[-1]}."
        return f"The word starts with {t[0]}."
    # Level 2 is the strongest hint — never reveals the word (matches game.py).
    correct = sum(1 for a, b in zip(d, t) if a == b) if len(d) == len(t) else 0
    praise = f"You already have {correct} letters in the right spot. " if correct else ""
    return f"{praise}It starts with {t[0]} and ends with {t[-1]}."


def _builtin_dynamic_lines() -> list[str]:
    lines: set[str] = set()

    # Prefixes the "wrong" handler glues in front of the tailored hint.
    # MUST mirror phrases.json's wrong_attempt1 / wrong_attempt2 lists —
    # the spoken line is f"{prefix} {hint}".
    _WRONG1_PREFIXES = PHRASES.get("wrong_attempt1", [])
    _WRONG2_PREFIXES = PHRASES.get("wrong_attempt2", [])

    # Streaks worth pre-caching; a 5+ streak is rare enough that a one-time
    # live-synth pause for that celebratory moment is a fine tradeoff for
    # the ~150-line savings of not caching every count up to 10.
    _STREAKS = range(2, 5)

    for target in _ANIMAL_WORDS:
        clue = _ANIMAL_CLUES.get(target.lower(), "")
        t = target.upper()

        # ── card-detected letter prompt (Step 2) — MUST match the
        #    card_detected handler byte-for-byte: alphabetical order,
        #    repeated letters kept, single string. ─────────────────────
        _card_letters = ", ".join(sorted(t))
        lines.add(
            f"Put the letters {_card_letters} in the playing area "
            f"and arrange them."
        )

        # ── correct (praise + optional streak + clue + next card) —
        # MUST match the "correct" handler's f-strings exactly.
        base = f"Woo hoo! {t}! You got it!"
        tail = (f" {clue}" if clue else "") + " Bring a new card to the playing area!"
        lines.add(base + tail)
        for s in _STREAKS:
            lines.add(f"{base} That's {s} in a row!{tail}")

        # ── word failed (reveal + clue + next card) — matches handler.
        failed = f"Ooh, tricky! It was {t}."
        if clue:
            failed += f" {clue}"
        failed += " No worries. New card!"
        lines.add(failed)

        lines.add(f"Oops! We did {t} already. Pick a new card!")

        # ── wrong attempt 1: prefix + level-1 hint ────────────────────────
        hint1 = _build_hint(target, 1, "")
        for p in _WRONG1_PREFIXES:
            lines.add(f"{p} {hint1}")

        # ── wrong attempt 2: prefix + level-2 hint (all praise variants) ──
        for correct_count in range(0, len(target) + 1):
            fake_detected = ("X" * correct_count) + t[correct_count:]
            hint2 = _build_hint(target, 2, fake_detected)
            for p in _WRONG2_PREFIXES:
                lines.add(f"{p} {hint2}")

        # ── idle-help hints (levels 1-2 only — a hint NEVER reveals the
        # word; the failed-word reveal is a separate line above) ─────────
        for level, det_variants in ((1, [""]),
                                    (2, [("X" * c) + t[c:]
                                         for c in range(0, len(target) + 1)])):
            for det in det_variants:
                h = _build_hint(target, level, det)
                lines.add(h if h.startswith("Here is a clue")
                          else "Here is a hint. " + h)

    # ── personalized auto-start welcomes (single player — the common case;
    # multi-player combos fall back to live synthesis) ─────────────────────
    if PLAYER_NAMES:
        for name in PLAYER_NAMES.values():
            lines.add(f"Hi {name}! Let's play! {_SCRAMBLE_RULES}")
        print(f"[INFO] Added {len(PLAYER_NAMES)} personalized welcome lines")
    else:
        print("[INFO] No player_names.json next to server — personalized "
              "welcomes will be synthesized live on first use")
    lines.add(_SCRAMBLE_RULES)   # no-name fallback welcome

    # New login-flow lines from game.py (spoken via say_dynamic):
    _NEW_RULES = ("Here's how we play! Pick a card and put it on the yellow card area. "
                  "Then I'll tell you which letters to grab. Let's go!")
    lines.add(_NEW_RULES)
    lines.add("Player one! Show me your name card to play!")
    for name in PLAYER_NAMES.values():
        # MUST match game.py _announce_new_players byte-for-byte (the old
        # mirror here had a trailing "Or we start in a few seconds." that
        # game.py doesn't send — a silent cache miss on every login).
        lines.add(f"Hi {name}! Now player two — show me your card!")
        lines.add(f"Hi {name}! Everyone's here. Let's play!")
        lines.add(f"Hi {name}!")
        # speed_game.py's "you go first" line, per name (its welcome is now
        # sent as components — see _show_welcome_screen).
        lines.add(f"{name}, you go first! Spell words from the tiles "
                  f"and press the buzzer for each one.")

    # Fixed tails for the component-style greetings. These are what let the
    # per-name parts above stay cacheable: N names x 1 clip, plus these,
    # instead of N^2 name-pair clips.
    lines.add("Everyone's here. Let's play!")
    lines.add("I got both cards!")
    lines.add(_SPEED_WELCOME_RULES)
    lines.add("Let's go!")
    # The rules line (_NEW_RULES, added above) is now sent name-free by
    # game.py — one fixed string, cached once. The old per-name
    # "Hi <name>! <rules>" variants and the never-cached duo
    # "Hi <A> and <B>! <rules>" case are gone.

    # Speed-game turn instructions — MUST match speed_game.py's round_start
    # `say` strings exactly. Personalized per name (fallback "Player N"
    # covers unregistered cards). Round length is 4 minutes
    # (ROUND_SECONDS=240 in speed_game.py) — keep in sync.
    _speed_names = list(PLAYER_NAMES.values()) + ["Player 1", "Player 2"]
    for n in _speed_names:
        lines.add("You have seven letter tiles. Make as many words as "
                  "you can! Each word needs three letters or more. "
                  "Press the buzzer for each word. "
                  f"{n}, your four minutes start now. Go!")
        lines.add(f"Your turn, {n}! Same game — make words with "
                  "your tiles and buzz each one. Four minutes. Go!")
    lines.add(_SPEED_RULES_P1)   # legacy strings, kept warm just in case
    lines.add(_SPEED_RULES_P2)

    # Pre-login "seated and ready?" invite. Sent as ready_prompt(say=...) by
    # BOTH speed_game.py's _show_ready_screen() and game.py's — one string,
    # one cache entry, byte-for-byte identical in all three files. It was
    # never cached before, so the very first kid of the day waited on a live
    # Piper synth for it.
    lines.add(_SEATED_LINE)

    lines.update(_WORD_FOUND_LINES)
    lines.add(_KEEP_GOING_LINE)

    # Time-remaining announcements. The scramble game (game.py) still uses the
    # minute-based boundaries; the speed game now uses short 60s turns and only
    # announces "Thirty seconds left!" and "Ten seconds left!" — both must be
    # pre-cached so the in-turn warning never pays a live-synthesis delay.
    lines.update([
        "Three and a half minutes left!",
        "Three minutes left!",
        "Two and a half minutes left!",
        "Two minutes left!",
        "One and a half minutes left!",
        "One minute left!",
        "Thirty seconds left!",
        "Ten seconds left!",
    ])

    lines.update([
        "Needs more letters!",
        "Already found that one!",
        "That's not a word!",
        "Those letters aren't in your pool!",
        "Try again!",
        "New card! Let's start that one instead.",
        "Looks right! Press the buzzer!",
    ])
    for player in (1, 2):
        lines.add(f"Time's up, Player {player}!")
        # 0-30 words covers a realistic range for a 4-minute round with this
        # word bank; anything beyond that is a rare outlier and falls back
        # to live synthesis rather than pre-caching scores nobody hits.
        for score in range(0, 31):
            word = "word" if score == 1 else "words"
            lines.add(f"Time's up, Player {player}! You found {score} {word}.")

    return sorted(lines)


def _all_lines():
    """Every phrase the robot might ever say — used to warm the cache.

    Combines the static phrases.json entries, the personalized name
    greetings, the built-in dynamic lines above (word reveals, clues, hints,
    speed-game feedback — baked directly into this file from the current
    word lists), and dynamic_lines.json if that file happens to exist
    (optional — see build_speech_manifest.py for a way to regenerate this
    from game.py/speed_game.py directly instead of keeping the copies above
    in sync by hand).
    """
    lines = []
    for plist in PHRASES.values():
        for s in plist:
            lines.append(s)

    lines.extend(all_greeting_lines())   # personalised name greetings

    builtin = _builtin_dynamic_lines()
    lines.extend(builtin)
    print(f"[INFO] Added {len(builtin)} built-in dynamic lines "
          "(word reveals, clues, hints, speed-game feedback)")

    if os.path.exists(DYNAMIC_LINES_FILE):
        try:
            with open(DYNAMIC_LINES_FILE, "r", encoding="utf-8") as f:
                dynamic = json.load(f)
            lines.extend(dynamic)
            print(f"[INFO] Loaded {len(dynamic)} additional dynamic lines "
                  f"from {DYNAMIC_LINES_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to load {DYNAMIC_LINES_FILE}: {e}")

    return sorted(set(lines))


def pregenerate():
    if not os.path.exists(PIPER_MODEL):
        print(f"[ERROR] model not found: {PIPER_MODEL}")
        return
    lines = _all_lines()
    print(f"[INFO] Pre-generating {len(lines)} clips (one-time)...")
    start = time.time()
    done = 0
    for i, line in enumerate(lines, 1):
        if gen(line):
            done += 1
        if i % 5 == 0 or i == len(lines):
            print(f"  {i}/{len(lines)} ({done} ready) [{time.time()-start:.0f}s]")
    print(f"[DONE] {done}/{len(lines)} cached in {time.time()-start:.0f}s.")


def main():
    import sys
    if "--pregenerate" in sys.argv:
        pregenerate()
        return

    # Connect + wake + gaze + motion BEFORE anything speaks.
    bot.start()

    router.start()

    # Warm the name-greeting cache before the robot begins speaking.
    # This avoids the first personalized greeting paying an on-demand Piper
    # synthesis delay when a card is detected and a name needs to be said.
    print("[INFO] Warming name greeting cache...")
    for line in all_greeting_lines():
        gen(line)
    print("[INFO] Name greeting cache is ready.")

    print("[INFO] Booting server... Launching camp introduction dialogue.")
    bot.gesture("celebrate")
    bot.say("intro")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Running on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopping")
        bot.stop()


if __name__ == "__main__":
    main()