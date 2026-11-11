"""
speed_game.py — Speed Word Builder (2-player, alternating turns)
===============================================================
Both players SHARE one 14-letter tile pool. They take short alternating turns
(60 s each, 4 turns per player = 8 minutes total). On your turn you spell as
many valid words as you can from the shared tiles; when your time is almost up
the robot warns you, then it's the other player's turn. NO WORD may be used
twice — not by you and not by your opponent — so the kid waiting can plan, but
whoever grabs a word first owns it. Most words at the end wins.

Physical letter tiles are read by the camera via custom ArUco markers. Player
ID cards use standard ArUco.

Flow: Scan P1 → Scan P2 → P1(60s) → P2(60s) → P1 → P2 → … (4 each) → Results

Keys:  SPACE / buzzer = submit word  ·  R = new game  ·  Q = quit

CHANGES (latest):
  - SHARED 14-letter pool (was: two separate 7-letter pools). Both players
    build from the same tiles on the table.
  - ALTERNATING TURNS: TURN_SECONDS (60s) each, TURNS_PER_PLAYER (4) turns
    apiece, TOTAL = 8 min. Keeps both kids engaged; no long idle waiting.
  - GLOBAL no-repeat: a word found by EITHER player is locked out for the
    rest of the game (gs["found_words"]).
  - Both players scan in ONCE up front; turns then alternate with a short
    handover screen (no re-scanning every turn).
  - Shared-pool dictionary loads from optional word_list.txt (drop it next
    to this file — see build_word_list.py to generate one); falls back to the
    built-in curated word set if that file is absent.
  - Robot time warnings at 30 s and 10 s remaining each turn.
  - LOGIN like game.py: showing a player ID card logs the player in
    (stable-frame confirmation) — no buzzer needed. SPACE/buzzer still works
    as a manual fallback on the scan screens.
  - DUAL-CAMERA ID detection: player cards are detected on BOTH the external
    game camera and the Reachy Mini's own camera (GET /player_ids).
  - Player names shown/spoken via player_names.json (get_player_name).
"""

import atexit
import collections
import json
import os
import platform
import random
import socket
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime
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

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    _XLSX_OK = True
except ImportError:
    _XLSX_OK = False
    print("openpyxl not available — logs will not be saved to .xlsx.")

from detection import (detect_markers, load_templates, create_marker_id_dict,
                       detect_player_ids, draw_player_markers,
                       detect_play_area, filter_markers_by_play_area,
                       draw_play_area, get_all_aruco_ids, hull_from_points)

# ── Player names (player_names.json) ─────────────────────────────────────────
try:
    from player_names import get_player_name, get_player_age, get_player_consent
except ImportError:
    def get_player_name(player_id):
        return "Player" if player_id is None else f"Player {player_id}"
    def get_player_age(player_id):
        return None
    def get_player_consent(player_id):
        return False
    print("[SPEED] player_names.py not found — using generic player labels.")

# ── Import shared word bank from game.py (avoids duplication) ─────────────────
try:
    from game import WORD_BANK
    print(f"[SPEED] Word bank loaded from game.py: {len(WORD_BANK)} words.")
except ImportError:
    print("[SPEED] game.py not found — using built-in fallback word bank.")
    WORD_BANK = []

# ─────────────────────────────────────────────────────────────────────────────
# SHARED LETTER POOL  (one 14-tile pool, used by BOTH players)
# 14 distinct high-frequency letters — one physical tile each, so every word
# may use each letter at most ONCE (no doubled letters).
# ─────────────────────────────────────────────────────────────────────────────
SHARED_POOL = list('ETAOINSHRDLUCP')   # E T A O I N S H R D L U C P

# Welcome rules block, spoken as one component of the post-login welcome.
# Deliberately name-free — that's what makes it a single cached clip rather
# than one per name pairing. Mirrored byte-for-byte in
# reachy_feedback_server.py as _SPEED_WELCOME_RULES; the audio cache is keyed
# on an md5 of the exact text, so any drift means a live synth.
# Pre-login "seated and ready?" invite. Byte-for-byte identical to game.py's
# _SEATED_SAY and reachy_feedback_server.py's _SEATED_LINE — one cached clip
# serves both games.
_SEATED_SAY = ("Hello! When you're seated and ready, press the buzzer. "
               "Then we'll get started.")

_SPEED_WELCOME_RULES = (
    "This is Speed Builder: each player gets one minute per turn, "
    "and there are eight turns total. "
    "Make as many words as you can from the shared 14-letter pool.")

# ── Built-in fallback word set ────────────────────────────────────────────────
# Used only if word_list.txt is not found next to this script. This is the
# union of the old per-half curated lists PLUS a batch of common words that
# span both halves, so the game is playable out of the box. For FULL coverage
# of everything formable from the 14 tiles, generate word_list.txt once with
# build_word_list.py (run it on the Mac) and copy it next to this file.
_FALLBACK_WORDS = frozenset([
    # from the old P1 half (OHIPLTS)
    'hip','his','hit','hop','hot','ish','lip','lit','lop','lot','ohs','oil',
    'opt','pit','pot','sip','sit','sop','sot','til','tip','top','tho','hilt',
    'hips','hits','hops','host','hots','lips','lisp','list','lops','lost','lots',
    'oils','opts','pith','pits','plot','posh','post','pots','ship','shop','shot',
    'silo','silt','slip','slit','slop','slot','soil','spit','spot','stop','this',
    'tips','toil','tops','tosh','hilts','hoist','pilot','plots','polis','posit',
    'spilt','split','spoil','toils','pilots','polish','spoilt',
    # from the old P2 half (RUDNACE)
    'ace','and','ane','arc','are','can','car','cud','cue','cur','dan','dar',
    'den','due','dun','ear','end','era','ran','red','rue','run','aced','acne',
    'acre','cane','card','care','cran','crud','curd','cure','dace','dare','darn',
    'dean','dear','duce','dune','dure','earn','near','nerd','nude','race','rand',
    'read','rend','rude','rune','urea','cadre','caned','caner','cared','cedar',
    'crane','crude','cured','dance','dared','dunce','raced','rance','under',
    'dancer','craned',
    # common words that SPAN both halves (only possible with the shared pool)
    'air','ail','aid','ale','ant','ape','apt','art','ash','ate','din','dip',
    'dirt','dish','doe','dot','dip','ear','eat','ehs','eta','hae','hair','hard',
    'hare','haul','head','heal','heap','hear','heat','held','hen','herd','hero',
    'hind','hire','hoard','hoe','hold','hole','home','hone','hope','horde','hose',
    'hound','hour','house','idea','ideal','idle','ion','iron','isle','lace','lad',
    'lade','laid','lain','lair','lance','land','lane','lard','lash','late','lead',
    'lean','leap','learn','lend','lid','lie','lien','line','lion','load','loan',
    'lode','lone','lore','loud','lucid','lunar','lure','made','maid','mail','main',
    'nail','naps','nard','near','neat','nice','node','nope','nose','note','oat',
    'ode','oral','ouch','ounce','pace','paced','pad','paid','pain','pair','pale',
    'pane','pant','pardon','parch','pare','part','pase','past','pat','path','pea',
    'peal','pear','peat','pech','pedal','pen','pend','peon','pert','pest','pet',
    'phone','pica','pie','pier','pine','plan','plane','plaid','plead','plod','plod',
    'poach','pod','pond','pone','pore','porch','pouch','pound','pour','pride','pried',
    'prod','prude','pure','purse','rail','rain','raise','rale','ranch','rant','rap',
    'rape','rapt','rash','rate','ratio','read','real','ream','reap','rein','rent',
    'reset','rest','rich','ride','rind','rinse','riot','ripe','rise','road','roan',
    'roil','role','rope','rose','round','route','sail','sale','sand','sane','sap',
    'sard','sat','sate','sauce','scald','scale','scan','scar','scare','scold','scone',
    'score','scour','scout','sea','seal','sear','seat','sedan','send','sent','shad',
    'shade','shale','shape','shard','share','shark','shear','shed','shin','shine',
    'shire','shoal','shone','shore','shorn','shout','shred','shun','sich','side',
    'sild','sine','sinh','sire','site','snare','snide','snore','snout','soap','soar',
    'sold','solid','sonar','sonic','sorb','sord','sore','sound','sour','south','spade',
    'span','spar','spare','spear','spend','spice','spider','spine','spire','spore',
    'sprain','spread','sprint','spud','spur','staid','stain','stair','stale','stand',
    'stare','starch','stead','steal','stein','stern','stir','stole','stone','store',
    'strain','strand','strap','strep','strip','stripe','strode','stud','sued','suet',
    'suit','sun','sundae','sup','sure','tacit','tad','tail','tain','talc','tale',
    'tan','tap','tape','tar','tardo','tare','taro','teal','tead','tear','ten','tend',
    'tepid','tern','thin','thine','third','thorn','those','thread','three','throne',
    'tide','tide','tied','tier','tile','tin','tine','tinder','tire','toad','toe',
    'toea','told','tone','toner','tonic','tore','torn','torso','tour','trace','traced',
    'trad','trade','trail','train','trance','tread','tried','trio','tripod','trod',
    'true','tsar','tun','tuna','tundra','tune','tuner','ulcer','undo','uneat','union',
    'unit','unite','unread','unrip','unroll','unshod','unsold','unsound','uphold',
    'upland','uproad','urchin','urn','used','user','usher',
])


# ── Blocked words ─────────────────────────────────────────────────────────────
# Words that ARE formable from the pool and ARE in the dictionary, but that we
# don't want a robot announcing to 6-12 year olds at camp. Applied to EVERY
# word source (word_list.txt and the built-in fallback), so a regenerated word
# list can never sneak one back in.
#
# EDIT THIS LIST FREELY — it's just a set of uppercase strings, matched exactly
# (not by substring, so innocent words like CORAL / THORN / SPICE / SCRAPE /
# PISTOL / HORNET are never caught by accident).
#
# Deliberately NOT blocked, because they're ordinary words a kid could fairly
# spell and rejecting them would feel broken: SNATCH, SCAT, LOIN, LUST, ORAL,
# POT, STONED, STRIP, SLOPE, SNIPER, URINE, URINATE, STUPID, HOAR, PRUDE.
# Move any of them down here if you'd rather they were rejected.
_BLOCKED_WORDS = frozenset([
    # vulgar
    'CRAP', 'CRAPS', 'SHIT', 'TURD', 'TURDS', 'ARSE',
    # slurs
    'CUNT', 'CUNTS', 'SPIC',
    # sexual / anatomy
    'ANUS', 'PENIS', 'SCROTA', 'SCROTAL', 'RECTA', 'RECTAL', 'RECTI', 'RECTUS',
    'EROTIC', 'EROTICA', 'EROTICAL', 'EROTICS', 'UNEROTIC', 'INCEST',
    'PORN', 'PORNS', 'PORNIEST', 'SLUT',
    'RAPE', 'RAPED', 'RAPES', 'RAPIST', 'RAPINE',
    'RAUNCH', 'RAUNCHES', 'RAUNCHIEST',
    'NUDE', 'NUDER', 'NUDES', 'NUDEST',
    'PANTIE', 'PANTIES', 'UNDIES', 'DOUCHE', 'DOUCHES', 'HORNIEST',
    'LECH', 'LECHES',
    # drugs
    'HEROIN', 'HEROINS', 'OPIATE', 'OPIATED', 'OPIATES', 'DOPER', 'DOPERS',
])


def _load_word_list():
    """Return an uppercase frozenset of valid words for the SHARED pool.

    Preference order:
      1. word_list.txt next to this script (one word per line). This is the
         real dictionary — ~7.5k words formable from the pool. Ship it.
      2. The built-in _FALLBACK_WORDS set (~500 words). Playable in a pinch,
         but it is MISSING obvious words like CAT and STAR, so kids will get
         told a valid word is invalid. A loud warning is printed if we land
         here — if you see it at camp, word_list.txt didn't get copied.

    Either way the result is filtered so every returned word is actually
    buildable from the 14 shared tiles (MIN_WORD_LENGTH..MAX_WORD_LENGTH
    letters, each letter used at most once) and is not in _BLOCKED_WORDS.
    """
    pool_counts = Counter(SHARED_POOL)

    def _formable(w):
        w = w.upper()
        if not (MIN_WORD_LENGTH <= len(w) <= MAX_WORD_LENGTH) or not w.isalpha():
            return False
        wc = Counter(w)
        return all(pool_counts.get(c, 0) >= n for c, n in wc.items())

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "word_list.txt")
    raw, source = None, ""
    try:
        with open(path, encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]
        source = f"word_list.txt ({len(raw)} lines)"
    except FileNotFoundError:
        print(f"[SPEED] *** WARNING: {path} not found. Falling back to the "
              f"built-in word set, which is missing common words (CAT, STAR, "
              f"CHAIR...). Kids WILL see valid words rejected. Copy "
              f"word_list.txt next to speed_game.py. ***")
    except Exception as e:
        print(f"[SPEED] *** WARNING: could not read {path} ({e}); "
              f"falling back to the built-in word set. ***")

    if raw is None:
        raw = list(_FALLBACK_WORDS)
        source = f"built-in fallback ({len(raw)} words)"

    words = frozenset(w.upper() for w in raw
                      if _formable(w) and w.upper() not in _BLOCKED_WORDS)
    print(f"[SPEED] Word source: {source}")
    print(f"[SPEED] {len(words)} words are formable from the shared pool "
          f"{''.join(SHARED_POOL)} at {MIN_WORD_LENGTH}-{MAX_WORD_LENGTH} "
          f"letters (after removing {len(_BLOCKED_WORDS)} blocked words).")
    return words


# NOTE: MIN_WORD_LENGTH must be defined before _load_word_list() runs; it is
# set in the CONFIG block below, so WORDS is populated there (search "WORDS =").

# ── Buzzer ─────────────────────────────────────────────────────────────────────
BUZZER_PIN    = 20
_buzzer_event = threading.Event()
try:
    from gpiozero import Button as _GPIOButton
    _buzzer_btn = _GPIOButton(BUZZER_PIN, pull_up=False, bounce_time=0.05)
    _buzzer_btn.when_pressed = _buzzer_event.set
    print(f"[SPEED] Buzzer ready on GPIO {BUZZER_PIN}.")
except Exception as _e:
    _buzzer_btn = None
    print(f"[SPEED] No GPIO buzzer ({_e}) — keyboard-only mode.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
REVERSE_READING     = True
REACHY_FEEDBACK_URL = "http://reachy-mini.local:8765/event"
REACHY_ENABLED      = True

# ── Login (game.py-style: show ID card → auto-start, no buzzer) ───────────────
LOGIN_STABLE_FRAMES = 4     # camera-loop frames an ID must persist before login
ROBOT_ID_POLL_S     = 0.30  # how often we poll the robot camera for IDs
ROBOT_ID_TTL_S      = 1.0   # robot-seen ID counts as "present" for this long

# ── Inactivity watchdog ───────────────────────────────────────────────────────
# If the ROBOT's camera sees no face AND the game camera sees no activity
# (no tiles, no cards, no buzzer) for this long, the game ends automatically
# with a goodbye. Applies once the game has started (P1's turn onward).
INACTIVITY_TIMEOUT_S = 90

# ── Turn structure ────────────────────────────────────────────────────────────
# Players SHARE one pool and take short alternating turns. Total game time is
# TURN_SECONDS * TURNS_PER_PLAYER * 2. Default: 60s * 4 * 2 = 480s = 8 minutes.
TURN_SECONDS      = 60    # length of each single turn
TURNS_PER_PLAYER  = 4     # how many turns each player gets
TOTAL_TURNS       = TURNS_PER_PLAYER * 2
HANDOVER_SECONDS  = 3     # brief "next player, get ready" pause between turns
WELCOME_DELAY_MS  = 1200   # wait before the first rotation starts so speech finishes
AUTO_RESTART_DELAY_S = 60  # restart automatically after results for next players

# Seconds-remaining marks at which the robot announces the time left in a turn.
# Only marks strictly below TURN_SECONDS are used. Each phrase MUST also exist,
# byte-for-byte, in reachy_feedback_server.py's pre-cache list (search
# _builtin_dynamic_lines) or it will be synthesized live the first time.
TURN_TIME_MARKS = (30, 10)


def _time_left_phrase(secs: int) -> str:
    """Kid-friendly spoken phrase for an in-turn time announcement.
    MUST stay in sync with the pre-cached copies in
    reachy_feedback_server.py's _builtin_dynamic_lines()."""
    return {
        30: "Thirty seconds left!",
        10: "Ten seconds left!",
    }.get(secs, f"{secs} seconds left!")


MIN_WORD_LENGTH     = 3     # fewest letters a submitted word may have
# Most letters the play area can PHYSICALLY hold. This is a hardware fact, not
# a game-design choice: only 4 tiles fit inside the boundary markers, so a
# longer word can never be built. Enforced in two places — the word list is
# filtered to <= this on load, and _submit_word rejects anything longer. A
# reading longer than this means a ghost tile got detected, not a real word,
# which is why it gets its own feedback message rather than "not in word list".
# If you enlarge the physical play area, bump this AND regenerate word_list.txt.
MAX_WORD_LENGTH     = 4

# ── Detection-loop cost controls ──────────────────────────────────────────────
# The "ArUco IDs seen: [...]" readout on the camera view is a setup/debug aid.
# Producing it costs TWO extra full ArUco passes (4x4 + 6x6) over every single
# frame, which directly slows how fast a tile change shows up on screen. Leave
# it False at camp; flip to True when you're aligning boundary markers and need
# to know which IDs the camera can actually see.
SHOW_ARUCO_IDS         = False
SHOW_ARUCO_IDS_EVERY_N = 10   # when on, refresh the ID list every Nth frame

# Frames to grab-and-discard before each real read, to clear anything the
# driver buffered while the previous detection pass was running. 2 suits a
# ~30 fps camera against a ~100 ms detection pass. If the view still feels
# behind your hand, raise to 3-4; if it feels jumpy or drops tiles, lower to 1.
CAM_DRAIN_FRAMES = 2
SUBMIT_COOLDOWN_S   = 1.2   # seconds before the exact same word can be re-submitted

# ── Shared-pool dictionary ────────────────────────────────────────────────────
# Populated now that MIN_WORD_LENGTH exists. WORDS is the authoritative valid-
# word set (uppercase), already filtered to what's buildable from SHARED_POOL.
WORDS = _load_word_list()

# ── Hands-away buzzer grace ───────────────────────────────────────────────────
# Kids press the buzzer with a hand/arm still over the tiles, so the camera
# reads an occluded word at the moment of the press. Stability alone proved
# NOT sufficient: a RESTING arm produces a partial reading that is perfectly
# steady frame-to-frame, so it passes a stability check while still covering
# tiles. Fix: after every press, wait a fixed HANDS_DELAY_S first (hands
# naturally retract right after pressing), and only THEN require a stable
# reading. The word is judged from that settled post-delay view.
HANDS_DELAY_S       = 2.0   # pause after the press before reading (slow path)
HANDS_STABLE_FRAMES = 3     # consecutive identical frames = clean view
HANDS_WAIT_MAX_S    = 3.0   # extra time after the delay to find a stable view
HANDS_POLL_MS       = 80    # how often the hands-wait poller re-checks
# Fast path: if the reading was ALREADY this stable at the moment of the press,
# the tiles were clearly untouched and we judge immediately with no delay. The
# bar is deliberately much higher than HANDS_STABLE_FRAMES — a resting arm can
# hold a partial reading still for 3 frames, but not for ~8 (≈0.5s+). Raise
# this if you still see arms judged as words; lower it for snappier feedback.
HANDS_FAST_STABLE_FRAMES = 8
FEEDBACK_DURATION   = 2.2   # seconds to show per-word feedback

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
P = {
    'bg':     '#0d0d0d',
    'panel':  '#141414',
    'card':   '#1c1c1c',
    'text':   '#d4d4d4',
    'dim':    '#484848',
    'sep':    '#242424',
    'green':  '#4ade80',
    'gold':   '#fbbf24',
    'red':    '#f87171',
    'blue':   '#60a5fa',
    'purple': '#a78bfa',
    'orange': '#fb923c',
    'p1':     '#38bdf8',   # sky blue  — Player 1
    'p2':     '#f97316',   # orange    — Player 2
    'p1_dim': '#0c2d3d',
    'p2_dim': '#3d1a00',
}


# ─────────────────────────────────────────────────────────────────────────────
# REACHY FEEDBACK CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class ReachyFeedbackClient:
    """Fire-and-forget JSON event sender to the Reachy Mini feedback server."""

    def __init__(self, url=REACHY_FEEDBACK_URL, enabled=REACHY_ENABLED):
        self.enabled       = enabled
        self.hostname_url  = url
        self.url           = url
        self.last_sent     = {}
        self.failures      = 0
        self._resolve()

    def _resolve(self):
        try:
            p  = urlparse(self.hostname_url)
            ip = socket.gethostbyname(p.hostname)
            self.url = f"{p.scheme}://{ip}:{p.port}{p.path}"
            self.failures = 0
        except Exception as exc:
            self.url = self.hostname_url
            print(f"[Reachy] not resolved ({exc})")

    def _ids_url(self):
        """URL of the robot's player-ID endpoint (…/player_ids)."""
        p = urlparse(self.url)
        return f"{p.scheme}://{p.netloc}/player_ids"

    def _busy_url(self):
        """URL of the robot's speech-busy endpoint (…/busy)."""
        p = urlparse(self.url)
        return f"{p.scheme}://{p.netloc}/busy"

    def get_presence(self):
        """Fetch presence info from the ROBOT's camera.

        Returns {"ids": [player ids], "face_seen_ago": secs-or-None}.
        Blocking (short timeout) — call from a background thread, never the
        GUI thread. Degrades to empty/None on any failure so the game falls
        back to external-camera-only detection.
        """
        if not self.enabled:
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

    def get_busy(self):
        """Return the robot speech status, or None on failure."""
        if not self.enabled:
            return None
        try:
            with urllib.request.urlopen(self._busy_url(), timeout=0.5) as r:
                data = json.loads(r.read().decode())
            return bool(data.get("busy", False))
        except Exception:
            return None

    def send(self, event_type, **payload):
        if not self.enabled:
            return
        now, cooldown = time.time(), 0.35
        if now - self.last_sent.get(event_type, 0) < cooldown:
            return
        self.last_sent[event_type] = now
        data = json.dumps({"type": event_type, **payload}).encode()

        def _post():
            try:
                req = urllib.request.Request(
                    self.url, data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=0.6).read()
                self.failures = 0
            except Exception:
                self.failures += 1
                if self.failures >= 3:
                    self._resolve()

        threading.Thread(target=_post, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# SPEED LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class SpeedLogger:
    """Records both players' turns to .xlsx + .json + .jsonl.

    Mirrors game_logger.SessionLogger's output set, which speed_game never
    had — it wrote xlsx only, so the speed sessions were missing from the
    machine-readable side of the study entirely.

    Three outputs per session:
      1. logs/speed_<ts>.xlsx        — human-readable, two sheets
      2. logs/speed_<ts>.json        — full mirror of this session
      3. logs/speed_master.jsonl     — one line per session, all sessions:
             import pandas as pd
             df = pd.read_json("logs/speed_master.jsonl", lines=True)

    save() can run more than once per session (game over, then again on
    quit), so the master is deduped by session_id — each session appears
    exactly once, holding its most recent data.

    Kept separate from sessions_master.jsonl on purpose: the two games have
    genuinely different shapes (scramble logs `puzzles` with attempts, speed
    logs `turns` with word lists), and merging them would give every row a
    half-empty set of columns. Both records carry a "game" field, so
    concatenating them later is still one line of pandas if you want it.
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.rounds     = []   # one entry per TURN

    def record_turn(self, player_num, player_id, pool, words, duration_s, turn_no):
        label = get_player_name(player_id) if player_id is not None \
            else f"Player {player_num}"
        self.rounds.append({
            'player_num': player_num,
            'player_id':  player_id,
            'player':     label,
            'turn_no':    turn_no,
            'pool':       ''.join(pool),
            'words':      list(words),
            'word_count': len(words),
            'duration_s': int(duration_s),
        })
        print(f"[LOG] Turn {turn_no} — {label}: {len(words)} words "
              f"in {duration_s:.0f}s — {words}")

    # Back-compat alias in case anything still calls the old name.
    def record_round(self, player_id, pool, words, duration_s):
        self.record_turn(0, player_id, pool, words, duration_s, 0)

    def _build_record(self, session_id, end_time):
        """Assemble the machine-readable mirror of this session."""
        elapsed_s = int((end_time - self.start_time).total_seconds())

        # Aggregate per player across their 4 turns.
        players = {}
        for r in self.rounds:
            key = r["player_num"]
            p = players.setdefault(key, {
                "num":         r["player_num"],
                "id":          r.get("player_id"),
                "name":        r["player"],
                "age":         get_player_age(r.get("player_id")),
                "consent":     get_player_consent(r.get("player_id")),
                "words_found": 0,
                "turns_taken": 0,
            })
            p["words_found"] += r["word_count"]
            p["turns_taken"] += 1
        players = [players[k] for k in sorted(players)]

        # Winner by word count. Solo sessions have no winner; equal counts tie.
        winner = None
        if len(players) == 2:
            a, b = players
            if a["words_found"] > b["words_found"]:
                winner = a["name"]
            elif b["words_found"] > a["words_found"]:
                winner = b["name"]
            else:
                winner = "tie"

        return {
            "game":             "speed_builder",
            "session_id":       session_id,
            "date":             self.start_time.strftime("%Y-%m-%d"),
            "start_time":       self.start_time.strftime("%H:%M:%S"),
            "end_time":         end_time.strftime("%H:%M:%S"),
            "duration_seconds": elapsed_s,
            "pool":             "".join(SHARED_POOL),
            "turn_seconds":     TURN_SECONDS,
            "turns_total":      TOTAL_TURNS,
            "turns_per_player": TURNS_PER_PLAYER,
            "players":          players,
            "winner":           winner,
            "total_words":      sum(r["word_count"] for r in self.rounds),
            "turns": [
                {
                    "turn_no":          r["turn_no"],
                    "player_num":       r["player_num"],
                    "player_id":        r.get("player_id"),
                    "player":           r["player"],
                    "pool":             r["pool"],
                    "duration_seconds": r["duration_s"],
                    "word_count":       r["word_count"],
                    "words":            [w.upper() for w in r["words"]],
                }
                for r in self.rounds
            ],
        }

    def _save_json(self, session_id, data, logd):
        json_path = os.path.join(logd, f"speed_{session_id}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[LOG] Saved → {json_path}")
        except Exception as e:
            print(f"[LOG] WARN: could not write {json_path}: {e}")

        master_path = os.path.join(logd, "speed_master.jsonl")
        try:
            records = []
            if os.path.exists(master_path):
                with open(master_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("session_id") != session_id:
                                records.append(rec)
                        except json.JSONDecodeError:
                            continue   # drop a corrupt line rather than crash
            records.append(data)
            tmp = master_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, master_path)   # atomic — a crash can't corrupt it
            print(f"[LOG] Master log updated → {master_path} "
                  f"({len(records)} sessions)")
        except Exception as e:
            print(f"[LOG] WARN: could not update {master_path}: {e}")

    def save(self):
        if not self.rounds:
            return None

        base  = os.path.dirname(os.path.abspath(__file__))
        logd  = os.path.join(base, "logs")
        os.makedirs(logd, exist_ok=True)
        ts    = self.start_time.strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(logd, f"speed_{ts}.xlsx")

        # JSON FIRST, and unconditionally. The xlsx path below needs openpyxl;
        # this doesn't. The old code returned early when openpyxl was missing,
        # which meant a machine without it silently logged NOTHING for the
        # whole session. The study data is the part that must survive.
        self._save_json(ts, self._build_record(ts, datetime.now()), logd)

        if not _XLSX_OK:
            print("[LOG] openpyxl missing — skipping xlsx save (JSON written).")
            return None

        def _hdr(ws, row, cols):
            for ci, h in enumerate(cols, 1):
                c = ws.cell(row=row, column=ci, value=h)
                c.font      = Font(bold=True, color="FFFFFF")
                c.fill      = PatternFill("solid", start_color="2F5496")
                c.alignment = Alignment(horizontal="center", vertical="center")

        def _autofit(ws):
            for col in ws.columns:
                mx = max((len(str(c.value)) for c in col if c.value), default=8)
                ws.column_dimensions[col[0].column_letter].width = mx + 4

        wb = Workbook()

        # Aggregate per player across all their turns (preserve first-seen
        # order: player 1 then player 2).
        agg = collections.OrderedDict()
        for r in self.rounds:
            key = r['player_num'] or r['player']
            a = agg.setdefault(key, {
                'player': r['player'], 'words': [], 'seconds': 0, 'turns': 0})
            a['words'].extend(r['words'])
            a['seconds'] += r['duration_s']
            a['turns']   += 1

        # ── Sheet 1: Summary (one row per player) ─────────────────────────
        ws = wb.active
        ws.title = "Summary"
        hdrs = ["Player", "Turns", "Words Found", "Word List", "Total Time"]
        _hdr(ws, 1, hdrs)

        totals = [len(a['words']) for a in agg.values()]
        for ri, a in enumerate(agg.values(), 2):
            secs = a['seconds']
            vals = [a['player'], a['turns'], len(a['words']),
                    ', '.join(w.upper() for w in a['words']),
                    f"{secs//60}m {secs%60}s"]
            for ci, v in enumerate(vals, 1):
                c = ws.cell(row=ri, column=ci, value=v)
                c.alignment = Alignment(horizontal="center", vertical="center",
                                        wrap_text=(ci == 4))

        # Highlight the winner's row (unless it's a tie).
        if len(totals) == 2 and totals[0] != totals[1]:
            wr = 2 + totals.index(max(totals))
            for ci in range(1, len(hdrs) + 1):
                ws.cell(row=wr, column=ci).fill = \
                    PatternFill("solid", start_color="C6EFCE")

        _autofit(ws)

        # ── Sheet 2: Turn-by-Turn ─────────────────────────────────────────
        tt = wb.create_sheet("Turn by Turn")
        _hdr(tt, 1, ["Turn", "Player", "Words This Turn", "Count", "Seconds"])
        for ri, r in enumerate(self.rounds, 2):
            vals = [r['turn_no'], r['player'],
                    ', '.join(w.upper() for w in r['words']),
                    r['word_count'], r['duration_s']]
            for ci, v in enumerate(vals, 1):
                c = tt.cell(row=ri, column=ci, value=v)
                c.alignment = Alignment(horizontal="center", vertical="center",
                                        wrap_text=(ci == 3))
        _autofit(tt)

        # ── Sheet 3: Word Detail ──────────────────────────────────────────
        wd = wb.create_sheet("Word Detail")
        _hdr(wd, 1, ["Player", "Turn", "Word #", "Word"])
        row = 2
        for r in self.rounds:
            for wi, w in enumerate(r['words'], 1):
                wd.cell(row=row, column=1, value=r['player'])
                wd.cell(row=row, column=2, value=r['turn_no'])
                wd.cell(row=row, column=3, value=wi)
                wd.cell(row=row, column=4, value=w.upper())
                row += 1
        _autofit(wd)

        wb.save(path)
        print(f"[LOG] Speed game saved → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# GAME PHASES
# ─────────────────────────────────────────────────────────────────────────────
PH_READY    = "ready"
PH_SCAN_P1  = "scan_p1"
PH_SCAN_P2  = "scan_p2"
PH_WELCOME  = "welcome"
PH_HANDOVER = "handover"   # brief "next player, get ready" pause between turns
PH_P1_TURN  = "p1_turn"
PH_P2_TURN  = "p2_turn"
PH_RESULTS  = "results"


def _fresh_state():
    return {
        "phase":               PH_READY,
        "p1_id":               None,
        "p2_id":               None,
        "p1_name":             None,
        "p2_name":             None,
        "ended_by_inactivity": False,

        # Shared pool — the same 14 tiles for both players, every turn.
        "pool":                list(SHARED_POOL),

        # Cumulative word lists, growing across each player's 4 turns.
        "p1_words":            [],
        "p2_words":            [],
        # Global lock: any word found by EITHER player (uppercase). Enforces
        # "no word twice, not even by the other player".
        "found_words":         set(),

        # Turn tracking. turn_no runs 1..TOTAL_TURNS; odd = P1, even = P2.
        "turn_no":             0,
        "active_player":       1,
        "active_words":        [],   # alias to the active player's cumulative list
        "turn_start":          None,
        "turn_baseline":       0,    # len(active_words) at the start of this turn

        "detected":            "",
        "last_subs":           {},
        "feedback":            None,
        "feedback_time":       0.0,
        "feedback_ok":         True,
        "time_up_fired":       False,
        "time_marks_left":     [],   # remaining seconds-marks to announce this turn
    }


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class SpeedGameGUI:

    def __init__(self, root):
        self.root = root
        root.title("Speed Word Builder")
        root.geometry("1280x800")
        root.configure(bg=P['bg'])
        root.minsize(1000, 660)

        self.gs     = _fresh_state()
        self.reachy = ReachyFeedbackClient()
        self.logger = SpeedLogger()

        self._cam_stop         = threading.Event()
        self._last_scanned_ids = []
        self._play_area_hull   = None
        self._templates        = None
        self._marker_id_dict   = None

        # ── Auto-login state (game.py-style stable-frame confirmation) ──────
        self._login_candidate  = None   # ID currently being confirmed
        self._login_count      = 0      # consecutive frames it has been seen
        self._login_fired      = False  # guards double-login per scan screen
        self._login_ack_sent   = False  # quick detection feedback has been spoken
        self._auto_restart_token = 0   # cancel pending auto-restart if new game starts

        # ── Robot-camera IDs: {id: last_seen_time}, filled by poll thread ───
        self._robot_ids        = {}
        self._robot_ids_lock   = threading.Lock()
        self._robot_face_ago   = None          # secs since robot saw a face
        self._robot_face_time  = 0.0           # local time of that reading

        # ── Inactivity watchdog ──────────────────────────────────────────────
        self._last_activity    = time.time()   # tiles/cards/buzzer/IDs seen

        # ── Hands-away buzzer grace state ────────────────────────────────────
        self._stable_read        = ""     # last camera reading
        self._stable_frames      = 0      # consecutive frames it has held
        self._hands_wait_active  = False  # currently waiting for hands to clear
        self._hands_wait_deadline = 0.0

        # ── Debug ArUco-ID overlay state (only used when SHOW_ARUCO_IDS) ─────
        self._dbg_frame     = 0
        self._dbg_aruco_ids = []

        root.bind('<space>', lambda _: self._on_space())
        root.bind('<r>',     lambda _: self._new_game())
        root.bind('<R>',     lambda _: self._new_game())
        root.bind('<q>',     lambda _: self._quit())
        root.bind('<Q>',     lambda _: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        threading.Thread(target=self._load_templates,  daemon=True).start()
        threading.Thread(target=self._camera_loop,     daemon=True).start()
        threading.Thread(target=self._robot_id_poller, daemon=True).start()

        self._build_chrome()
        self._show_ready_screen()
        self._poll_buzzer()
        self._tick()

        # Tell Reachy which game is running so it can give the right welcome
        self.root.after(500, lambda: self.reachy.send("game_start", mode="speed"))

    def _quit(self):
        # Guard the logger save: if it ever raises, we must still reach
        # root.destroy() below, otherwise mainloop() never returns and the
        # recorder.stop() in main()'s `finally` block never runs either —
        # silently losing the whole session's video, not just the log.
        try:
            self.logger.save()
        except Exception as e:
            print(f"[LOG] WARN: save-on-quit failed ({e}); "
                  f"continuing shutdown so the recording still finalizes.")
        self._cam_stop.set()
        self.root.destroy()

    def _new_game(self):
        self._auto_restart_token += 1
        self.gs     = _fresh_state()
        self.logger = SpeedLogger()
        self._show_ready_screen()
        self._status("New game — press the buzzer when you're seated and ready")

    def _maybe_auto_restart(self, token):
        if token != self._auto_restart_token:
            return
        if self.gs.get("phase") == PH_RESULTS:
            self._new_game()

    # ── Template loading ──────────────────────────────────────────────────────

    def _load_templates(self):
        try:
            t = load_templates()
            d = create_marker_id_dict()
            self._templates      = t
            self._marker_id_dict = d
            self.root.after(0, lambda: self._status("Templates ready"))
        except Exception as e:
            self.root.after(0, lambda: self._status(f"Template error: {e}"))

    # ── Robot camera: player-ID polling ──────────────────────────────────────

    def _robot_id_poller(self):
        """Poll the Reachy server's /player_ids endpoint.

        Fast (ROBOT_ID_POLL_S) during scan phases so login feels instant;
        slower (1 s) during turns, where we only need the robot's
        face-presence signal for the inactivity watchdog. The robot scans its
        own camera for ArUco player cards (IDs 200-999) and reports them; we
        timestamp each sighting so an ID stays 'present' for ROBOT_ID_TTL_S
        even between polls.
        """
        while not self._cam_stop.is_set():
            scanning = self.gs.get("phase") in (PH_SCAN_P1, PH_SCAN_P2)
            info = self.reachy.get_presence()
            now  = time.time()
            if info["ids"]:
                with self._robot_ids_lock:
                    for pid in info["ids"]:
                        self._robot_ids[pid] = now
            self._robot_face_ago  = info["face_seen_ago"]
            self._robot_face_time = now
            time.sleep(ROBOT_ID_POLL_S if scanning else 1.0)

    def _robot_sees_face_within(self, window_s):
        """True if the ROBOT's camera saw a face within the last window_s.

        Unknown (robot offline / never saw a face) counts as False so the
        inactivity watchdog falls back to game-camera activity alone.
        """
        ago = self._robot_face_ago
        if ago is None:
            return False
        # Age the reading by the time since we polled it
        ago += max(0.0, time.time() - self._robot_face_time)
        return ago <= window_s

    def _current_robot_ids(self):
        """IDs the robot camera has seen within the last ROBOT_ID_TTL_S."""
        cutoff = time.time() - ROBOT_ID_TTL_S
        with self._robot_ids_lock:
            return [pid for pid, t in self._robot_ids.items() if t >= cutoff]

    # ── Auto-login (game.py-style: show card → start, no buzzer) ─────────────

    def _track_login(self, merged_ids, player_num):
        """Stable-frame confirmation, mirroring game.py's card logic.

        Called from the camera loop each scan frame. An ID must persist for
        LOGIN_STABLE_FRAMES consecutive frames before the turn auto-starts —
        avoids false triggers from a card waved past either camera. During
        the P2 scan, P1's card is ignored so one child can't log in twice.
        """
        if self._login_fired:
            return

        used = self.gs.get("p1_id") if player_num == 2 else None
        candidates = [pid for pid in merged_ids if pid != used]

        if not candidates:
            self._login_candidate = None
            self._login_count     = 0
            return

        pid = candidates[0]
        if pid == self._login_candidate:
            self._login_count += 1
        else:
            self._login_candidate = pid
            self._login_count     = 1
            self._login_ack_sent  = False

        self._give_login_feedback(player_num, pid)

        if self._login_count >= LOGIN_STABLE_FRAMES:
            self._login_fired = True
            self.root.after(0, lambda p=pid, n=player_num: self._auto_login(n, p))

    def _give_login_feedback(self, player_num, player_id):
        if self._login_ack_sent or self._login_candidate != player_id:
            return

        self._login_ack_sent = True
        self._status(f"Player {player_num} card detected")
        self.reachy.send("login_detected",
                         player=player_id,
                         say=f"Player {player_num} card detected.")

    def _auto_login(self, player_num, player_id):
        """GUI-thread handler: a stable player ID was confirmed → register the
        player and move the scan flow forward (P1 → scan P2 → begin rotation)."""
        expected = PH_SCAN_P1 if player_num == 1 else PH_SCAN_P2
        if self.gs.get("phase") != expected:
            return
        name = get_player_name(player_id)
        print(f"[LOGIN] {name} (ID {player_id}) logged in on scan {player_num}")
        if player_num == 1:
            status = f"Player 1 detected: {name}"
        else:
            status = f"Player 2 detected: {name}"
        self.reachy.send("player_detected", player=player_id)
        self._status(status)
        self._register_player(player_num, player_id)

    def _reset_login_tracking(self):
        self._login_candidate = None
        self._login_count     = 0
        self._login_fired     = False
        self._login_ack_sent  = False
        self._last_activity   = time.time()   # fresh watchdog window
        with self._robot_ids_lock:
            self._robot_ids.clear()

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self):
        from camera_select import detection_index
        cam_index = detection_index()
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.root.after(0, lambda: self._status("No camera found"))
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  900)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 560)
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
            # Drain whatever the driver buffered while we were busy detecting,
            # so `frame` is the CURRENT view of the table rather than a stale
            # one. grab() only decodes the header, so this is cheap.
            for _ in range(CAM_DRAIN_FRAMES):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue

            phase = self.gs["phase"]

            if phase in (PH_SCAN_P1, PH_SCAN_P2):
                # Local (external) camera detection…
                ids = detect_player_ids(frame)
                # …merged with IDs the ROBOT's camera has seen recently.
                robot_ids = self._current_robot_ids()
                merged = ids + [p for p in robot_ids if p not in ids]
                self._last_scanned_ids = merged
                if merged:
                    self._last_activity = time.time()

                # Auto-login: stable ID on either camera starts the turn.
                self._track_login(merged, 1 if phase == PH_SCAN_P1 else 2)

                disp = draw_player_markers(frame.copy(), ids)
                bpts, hull = detect_play_area(frame)
                if hull is not None:
                    self._play_area_hull = hull
                draw_play_area(disp, self._play_area_hull, bpts,
                               get_all_aruco_ids(frame))
                self.root.after(0, lambda f=disp, d=ids: self._on_scan_frame(f, d))

            elif phase in (PH_P1_TURN, PH_P2_TURN):
                disp     = frame.copy()
                detected = ""

                if self._templates is not None:
                    pool = self.gs["active_pool"]
                    markers, bpts = detect_markers(
                        frame, self._templates, self._marker_id_dict,
                        allowed_labels=set(pool))

                    # Only update the hull when ALL 4 boundary markers are
                    # visible — prevents flicker/shrink when one corner is
                    # briefly occluded (see same fix in game.py).
                    if len(bpts) >= 4:
                        h = hull_from_points(bpts)
                        if h is not None:
                            self._play_area_hull = h

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

                    # Overlay only — `bpts` already came back from
                    # detect_markers() above, so the old extra
                    # detect_play_area(frame) call here was a second full
                    # 4x4 ArUco pass over the same frame for data we already
                    # had. The ArUco-ID debug readout is a THIRD and FOURTH
                    # pass (4x4 + 6x6), so it now only runs when explicitly
                    # switched on, and only every SHOW_ARUCO_IDS_EVERY_N
                    # frames even then.
                    dbg_ids = None
                    if SHOW_ARUCO_IDS:
                        self._dbg_frame += 1
                        if self._dbg_frame % SHOW_ARUCO_IDS_EVERY_N == 0:
                            self._dbg_aruco_ids = get_all_aruco_ids(frame)
                        dbg_ids = self._dbg_aruco_ids
                    draw_play_area(disp, self._play_area_hull, bpts, dbg_ids)

                # Stability tracking for the hands-away buzzer grace: count
                # how many consecutive frames the reading has been identical.
                if detected == self._stable_read:
                    self._stable_frames += 1
                else:
                    self._stable_read   = detected
                    self._stable_frames = 1

                self.gs["detected"] = detected
                if detected:
                    self._last_activity = time.time()
                self.root.after(0,
                    lambda f=disp, w=detected: self._on_turn_frame(f, w))

        cap.release()

    # ── Frame callbacks (GUI thread) ──────────────────────────────────────────

    def _on_scan_frame(self, frame, ids):
        self._draw_on_canvas(getattr(self, '_cam_canvas', None), frame)
        self._update_scan_labels(ids)

    def _on_turn_frame(self, frame, detected):
        self._draw_on_canvas(getattr(self, '_cam_canvas', None), frame)
        self._update_detected_label(detected)

    def _draw_on_canvas(self, canvas, frame):
        if not _PIL_OK or canvas is None:
            return
        try:
            if not canvas.winfo_exists():
                return
            w     = max(canvas.winfo_width(),  100)
            h     = max(canvas.winfo_height(), 60)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo
        except Exception:
            pass

    # ── Persistent chrome ─────────────────────────────────────────────────────

    def _build_chrome(self):
        hdr = tk.Frame(self.root, bg=P['panel'], height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="SPEED  BUILDER",
                 font=("Helvetica", 17, "bold"),
                 bg=P['panel'], fg=P['gold']).pack(side=tk.LEFT, padx=20)

        self._lbl_phase = tk.Label(hdr, text="",
                                   font=("Helvetica", 11),
                                   bg=P['panel'], fg=P['dim'])
        self._lbl_phase.pack(side=tk.LEFT, padx=16)

        self._lbl_timer = tk.Label(hdr, text="",
                                   font=("Helvetica", 22, "bold"),
                                   bg=P['panel'], fg=P['gold'])
        self._lbl_timer.pack(side=tk.RIGHT, padx=24)

        self._lbl_scores = tk.Label(hdr, text="",
                                    font=("Helvetica", 11),
                                    bg=P['panel'], fg=P['text'])
        self._lbl_scores.pack(side=tk.RIGHT, padx=16)

        tk.Button(hdr, text="New Game",
                  command=self._new_game,
                  bg='#1d4ed8', fg='white',
                  font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                  activebackground='#1d4ed8', activeforeground='white',
                  ).pack(side=tk.RIGHT, padx=4)

        tk.Frame(self.root, bg=P['sep'], height=1).pack(fill=tk.X)

        self._lbl_status = tk.Label(
            self.root, text="Starting up...",
            font=("Helvetica", 9), bg=P['bg'], fg=P['dim'])
        self._lbl_status.pack(side=tk.BOTTOM, anchor=tk.W, padx=14, pady=3)

        self._stage = tk.Frame(self.root, bg=P['bg'])
        self._stage.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 0))

    def _status(self, msg):
        try:
            if self._lbl_status.winfo_exists():
                self._lbl_status.config(text=msg)
        except Exception:
            pass

    def _clear_stage(self):
        for w in self._stage.winfo_children():
            w.destroy()

    def _update_score_banner(self):
        phase = self.gs["phase"]
        s1    = len(self.gs["p1_words"])
        s2    = len(self.gs["p2_words"])
        # Before anyone has scored (the two scan screens) keep it clean;
        # once the rotation starts, always show both running totals so both
        # kids can see who's ahead.
        if phase in (PH_SCAN_P1, PH_SCAN_P2) and s1 == 0 and s2 == 0:
            self._lbl_scores.config(text="")
        else:
            p1n = self.gs.get("p1_name") or "P1"
            p2n = self.gs.get("p2_name") or "P2"
            self._lbl_scores.config(
                text=f"{p1n}: {s1}   |   {p2n}: {s2}", fg=P['text'])

    # ── Scan screen ───────────────────────────────────────────────────────────

    def _show_ready_screen(self):
        self._clear_stage()
        self._reset_login_tracking()
        self.gs["phase"] = PH_READY
        self._lbl_phase.config(text="REACHY  —  READY?", fg=P['gold'])
        self._lbl_timer.config(text="")
        self._update_score_banner()

        left = tk.Frame(self._stage, bg='black')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._cam_canvas = tk.Canvas(left, bg='black', highlightthickness=0)
        self._cam_canvas.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(self._stage, bg=P['bg'], width=440)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)

        mid = tk.Frame(right, bg=P['bg'])
        mid.pack(expand=True)

        tk.Label(mid, text="Hello!", font=("Helvetica", 32, "bold"),
                 bg=P['bg'], fg=P['gold']).pack(pady=(0, 12))
        tk.Label(mid,
                 text=("When you're seated and ready, press the buzzer. "
                       "Then we'll get started."),
                 font=("Helvetica", 13),
                 bg=P['bg'], fg=P['text'], justify=tk.CENTER).pack(pady=(0, 20))

        tk.Label(mid, text="Press SPACE or the buzzer when you're ready.",
                 font=("Helvetica", 12, "bold"),
                 bg=P['bg'], fg=P['green']).pack(pady=(0, 30))

        tk.Label(mid, text="After you press the buzzer, I will scan your cards.",
                 font=("Helvetica", 11),
                 bg=P['bg'], fg=P['dim'], justify=tk.CENTER).pack(pady=(0, 18))

        tk.Button(mid, text="Start scan", command=lambda: self._show_scan_screen(1),
                  bg=P['green'], fg='#000000', font=("Helvetica", 12, "bold"),
                  relief=tk.FLAT, padx=24, pady=10, cursor="hand2",
                  activebackground=P['green'], activeforeground='#000').pack(pady=(12, 0))

        self._status("Press SPACE or the buzzer when you're seated and ready")
        # seated_prompt, NOT ready_prompt: ready_prompt now means "rules are
        # playing, everyone is logged in" and flips the server to the playing
        # phase, which would stop face_greet greeting walk-ups at this screen.
        #
        # This is also a behaviour change worth knowing about: seated_prompt is
        # a silent declaration. The robot no longer announces this line the
        # instant the screen is built — it speaks it via face_greet only once
        # its camera actually sees somebody. speed_game auto-restarts every 60s
        # after results, so the old unconditional version was announcing
        # "when you're seated and ready" to an empty table all afternoon.
        self.reachy.send("seated_prompt", say=_SEATED_SAY)

    def _show_scan_screen(self, player_num):
        self._clear_stage()
        self._reset_login_tracking()          # fresh login state per scan screen
        self.gs["phase"] = PH_SCAN_P1 if player_num == 1 else PH_SCAN_P2
        accent = P['p1'] if player_num == 1 else P['p2']

        # Robot asks the child to show their name card (phrases: login_prompt)
        self.reachy.send("login_prompt", player=player_num)

        self._lbl_phase.config(text=f"PLAYER {player_num}  —  SCAN", fg=accent)
        self._lbl_timer.config(text="")
        self._update_score_banner()

        left = tk.Frame(self._stage, bg='black')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._cam_canvas = tk.Canvas(left, bg='black', highlightthickness=0)
        self._cam_canvas.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(self._stage, bg=P['bg'], width=440)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)

        mid = tk.Frame(right, bg=P['bg'])
        mid.pack(expand=True)

        badge = tk.Frame(mid, bg=accent, width=90, height=90)
        badge.pack(pady=(0, 18))
        badge.pack_propagate(False)
        tk.Label(badge, text=str(player_num),
                 font=("Helvetica", 48, "bold"),
                 bg=accent, fg='#0d0d0d').pack(expand=True)

        tk.Label(mid, text=f"PLAYER  {player_num}",
                 font=("Helvetica", 22, "bold"),
                 bg=P['bg'], fg=accent).pack(pady=(0, 6))

        if player_num == 2:
            p1n = self.gs.get("p1_name") or "Player 1"
            tk.Label(mid, text=f"{p1n} is in!  Player 2, log in to begin.",
                     font=("Helvetica", 12),
                     bg=P['bg'], fg=P['dim']).pack(pady=(0, 16))

        # Both players share ONE 14-tile pool — show it so they can lay it out.
        pool_str = "  ".join(SHARED_POOL)
        tk.Label(mid, text="Shared tiles:",
                 font=("Helvetica", 11), bg=P['bg'], fg=P['dim']).pack()
        tk.Label(mid, text=pool_str,
                 font=("Courier", 15, "bold"),
                 bg=P['bg'], fg=accent).pack(pady=(0, 12))

        tk.Label(mid,
                 text="Show your player ID card to either camera to log in.\n"
                      "(No buzzer needed to log in)",
                 font=("Helvetica", 13),
                 bg=P['bg'], fg=P['text'],
                 justify=tk.CENTER).pack(pady=(0, 20))

        self._lbl_scan_detected = tk.Label(mid, text="Scanning for card...",
                                           font=("Helvetica", 12, "bold"),
                                           bg=P['bg'], fg=P['dim'])
        self._lbl_scan_detected.pack(pady=(0, 20))

        tk.Frame(mid, bg=P['sep'], height=1).pack(fill=tk.X, pady=(0, 16))

        tk.Label(mid,
                 text=f"You'll take turns: {TURN_SECONDS} seconds each, "
                      f"{TURNS_PER_PLAYER} turns apiece.\n"
                      "Spell words from the shared tiles and press SPACE.\n"
                      "No word can be used twice — most words wins!",
                 font=("Helvetica", 10),
                 bg=P['bg'], fg=P['dim'],
                 justify=tk.CENTER).pack()

        tk.Button(mid,
                  text=f"Skip — play without Player {player_num} ID",
                  command=lambda: self._register_player(player_num, None),
                  bg='#2a2a2a', fg=P['dim'],
                  font=("Helvetica", 9),
                  relief=tk.FLAT, padx=14, pady=5, cursor="hand2",
                  ).pack(pady=(18, 0))

        self._status(
            f"Player {player_num}: show your ID card to the game camera or "
            f"Reachy to log in")

    def _update_scan_labels(self, ids):
        lbl = getattr(self, '_lbl_scan_detected', None)
        if not (lbl and lbl.winfo_exists()):
            return
        if ids:
            names = "  +  ".join(get_player_name(p) for p in ids)
            lbl.config(text=f"Detected:  {names}", fg=P['green'])
        else:
            lbl.config(text="Scanning for card...", fg=P['dim'])

    # ── Turn screen ───────────────────────────────────────────────────────────

    def _show_turn_screen(self, player_num):
        self._clear_stage()
        phase  = PH_P1_TURN if player_num == 1 else PH_P2_TURN
        accent = P['p1']    if player_num == 1 else P['p2']
        pool   = self.gs["pool"]           # shared 14 tiles
        pround = (self.gs["turn_no"] + 1) // 2

        self.gs["phase"] = phase
        self._lbl_phase.config(
            text=f"PLAYER {player_num}  —  GO!   "
                 f"(turn {pround}/{TURNS_PER_PLAYER})", fg=accent)
        self._update_score_banner()

        left = tk.Frame(self._stage, bg=P['panel'], width=340)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=4)
        left.pack_propagate(False)

        tk.Label(left, text="SHARED TILES",
                 font=("Helvetica", 9, "bold"),
                 bg=P['panel'], fg=accent).pack(pady=(14, 8))

        pool_frame = tk.Frame(left, bg=P['panel'])
        pool_frame.pack(padx=12, pady=(0, 12))
        for i, letter in enumerate(pool):
            tile = tk.Frame(pool_frame, bg=accent, width=50, height=50)
            tile.grid(row=i // 7, column=i % 7, padx=3, pady=3)
            tile.pack_propagate(False)
            tk.Label(tile, text=letter.upper(),
                     font=("Courier", 20, "bold"),
                     bg=accent, fg='#0d0d0d').pack(expand=True)

        remaining_words = len(WORDS) - len(self.gs["found_words"])
        tk.Label(left, text=f"{remaining_words} words still available",
                 font=("Helvetica", 8),
                 bg=P['panel'], fg=P['dim']).pack(pady=(0, 6))

        tk.Frame(left, bg=P['sep'], height=1).pack(fill=tk.X, padx=12, pady=(0, 8))

        lhdr = tk.Frame(left, bg=P['panel'])
        lhdr.pack(fill=tk.X, padx=14, pady=(0, 4))
        tk.Label(lhdr, text="YOUR WORDS",
                 font=("Helvetica", 9, "bold"),
                 bg=P['panel'], fg=P['dim']).pack(side=tk.LEFT)
        self._lbl_word_count = tk.Label(lhdr, text="0",
                                        font=("Helvetica", 11, "bold"),
                                        bg=P['panel'], fg=accent)
        self._lbl_word_count.pack(side=tk.RIGHT)

        self._words_frame = tk.Frame(left, bg=P['panel'])
        self._words_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        right = tk.Frame(self._stage, bg=P['panel'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=4)

        cam_wrap = tk.Frame(right, bg='black', height=310)
        cam_wrap.pack(fill=tk.X, padx=8, pady=(8, 0))
        cam_wrap.pack_propagate(False)
        self._cam_canvas = tk.Canvas(cam_wrap, bg='black', highlightthickness=0)
        self._cam_canvas.pack(fill=tk.BOTH, expand=True)

        det_row = tk.Frame(right, bg=P['panel'])
        det_row.pack(fill=tk.X, padx=16, pady=(12, 0))
        tk.Label(det_row, text="CAMERA READS:",
                 font=("Helvetica", 9, "bold"),
                 bg=P['panel'], fg=P['dim']).pack(side=tk.LEFT)
        self._lbl_detected = tk.Label(det_row,
                                      text="— arrange tiles —",
                                      font=("Courier", 24, "bold"),
                                      bg=P['panel'], fg=P['dim'])
        self._lbl_detected.pack(side=tk.LEFT, padx=14)

        self._lbl_match = tk.Label(right, text="",
                                   font=("Helvetica", 11, "bold"),
                                   bg=P['panel'])
        self._lbl_match.pack(pady=(2, 0))

        tk.Frame(right, bg=P['sep'], height=1).pack(fill=tk.X, padx=16, pady=8)

        self._lbl_feedback = tk.Label(right, text="",
                                      font=("Helvetica", 18, "bold"),
                                      bg=P['panel'])
        self._lbl_feedback.pack(pady=(4, 4))

        tk.Label(right,
                 text="SPACE = submit word  ·  R = new game  ·  Q = quit",
                 font=("Helvetica", 9),
                 bg=P['panel'], fg=P['dim']).pack()

        self._refresh_words_panel()
        pname = self.gs.get(f"p{player_num}_name") or f"Player {player_num}"
        self._status(
            f"{pname}: spell words from the shared tiles and press SPACE! "
            f"({TURN_SECONDS}s)")

    def _update_detected_label(self, detected):
        ld = getattr(self, '_lbl_detected', None)
        lm = getattr(self, '_lbl_match',    None)
        if not (ld and ld.winfo_exists()):
            return

        if not detected:
            ld.config(text="— arrange tiles —", fg=P['dim'])
            if lm and lm.winfo_exists():
                lm.config(text="")
            return

        word = detected.upper()
        pool = self.gs["pool"]
        pc   = Counter(c.upper() for c in pool)
        wc   = Counter(word)

        in_pool   = all(pc[c] >= n for c, n in wc.items())
        in_dict   = word in WORDS
        long_ok   = MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH
        already   = word in self.gs["found_words"]   # taken by EITHER player

        if in_pool and in_dict and long_ok and not already:
            ld.config(text=word, fg=P['green'])
            if lm and lm.winfo_exists():
                lm.config(text="✓  Valid word — press SPACE!", fg=P['green'])
        elif already:
            ld.config(text=word, fg=P['gold'])
            if lm and lm.winfo_exists():
                lm.config(text="Already taken — try another!", fg=P['gold'])
        elif len(word) > MAX_WORD_LENGTH:
            # Can't happen physically — only MAX_WORD_LENGTH tiles fit — so
            # this is a ghost tile rather than the kid overreaching.
            ld.config(text=word, fg=P['red'])
            if lm and lm.winfo_exists():
                lm.config(text="Too many letters — clear extra tiles!",
                          fg=P['red'])
        elif in_pool and long_ok and not in_dict:
            ld.config(text=word, fg=P['orange'])
            if lm and lm.winfo_exists():
                lm.config(text="Not in word list", fg=P['dim'])
        elif not in_pool and len(word) > 0:
            ld.config(text=word, fg=P['red'])
            if lm and lm.winfo_exists():
                lm.config(text="Letter not on the table!", fg=P['red'])
        else:
            ld.config(text=word, fg=P['dim'])
            if lm and lm.winfo_exists():
                lm.config(text="")

    def _refresh_words_panel(self):
        wf = getattr(self, '_words_frame', None)
        if not (wf and wf.winfo_exists()):
            return
        for w in wf.winfo_children():
            w.destroy()

        words  = self.gs["active_words"]
        phase  = self.gs["phase"]
        accent = P['p1'] if phase == PH_P1_TURN else P['p2']

        for i, word in enumerate(reversed(words)):
            idx = len(words) - i
            row = tk.Frame(wf, bg=P['card'], pady=5)
            row.pack(fill=tk.X, pady=1)
            tk.Frame(row, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
            tk.Label(row, text=f"{idx:02d}",
                     font=("Courier", 11, "bold"),
                     bg=P['card'], fg=P['dim'], width=3).pack(side=tk.LEFT, padx=(8, 4))
            tk.Label(row, text=word.upper(),
                     font=("Courier", 14, "bold"),
                     bg=P['card'], fg=P['text']).pack(side=tk.LEFT)
            tk.Label(row, text=f"+{len(word)}",
                     font=("Helvetica", 9),
                     bg=P['card'], fg=accent).pack(side=tk.RIGHT, padx=8)

        wc = getattr(self, '_lbl_word_count', None)
        if wc and wc.winfo_exists():
            wc.config(text=str(len(words)))

    # ── Results screen ────────────────────────────────────────────────────────

    def _show_results_screen(self):
        self._clear_stage()
        self.gs["phase"] = PH_RESULTS
        self._lbl_phase.config(text="FINAL  RESULTS", fg=P['gold'])
        self._lbl_timer.config(text="")
        self._update_score_banner()

        p1w, p2w = self.gs["p1_words"], self.gs["p2_words"]
        s1,  s2  = len(p1w), len(p2w)
        p1n = self.gs.get("p1_name") or "Player 1"
        p2n = self.gs.get("p2_name") or "Player 2"

        if s1 > s2:
            banner_text, banner_fg = f"{p1n.upper()}  WINS!", P['p1']
        elif s2 > s1:
            banner_text, banner_fg = f"{p2n.upper()}  WINS!", P['p2']
        else:
            banner_text, banner_fg = "IT'S  A  TIE!", P['gold']

        self.reachy.send("gameover", p1_score=s1, p2_score=s2,
                         winner=1 if s1>s2 else 2 if s2>s1 else 0)

        # Goodbye: robot announces final score with names, then a farewell
        # ("It was so fun playing with you!"). p2_name is None if P2 never
        # played (e.g. inactivity ended the game mid-way) — the robot then
        # announces a solo score instead.
        self.reachy.send(
            "goodbye", mode="speed",
            p1_name=self.gs.get("p1_name") or "Player one", p1_score=s1,
            p2_name=self.gs.get("p2_name"),                 p2_score=s2,
            reason=("inactivity" if self.gs.get("ended_by_inactivity")
                    else "finished"))

        top = tk.Frame(self._stage, bg=P['panel'])
        top.pack(fill=tk.X, pady=(0, 10))
        tk.Label(top, text=banner_text,
                 font=("Helvetica", 30, "bold"),
                 bg=P['panel'], fg=banner_fg).pack(pady=(14, 4))
        # Shared pool shown once (both players used the same tiles).
        pool_row = tk.Frame(top, bg=P['panel'])
        pool_row.pack(pady=(0, 12))
        for ltr in self.gs["pool"]:
            t = tk.Frame(pool_row, bg=P['dim'], width=26, height=26)
            t.pack(side=tk.LEFT, padx=1)
            t.pack_propagate(False)
            tk.Label(t, text=ltr.upper(), font=("Courier", 10, "bold"),
                     bg=P['dim'], fg='#0d0d0d').pack(expand=True)

        cols = tk.Frame(self._stage, bg=P['bg'])
        cols.pack(fill=tk.BOTH, expand=True)

        for pnum, (name, words, score, accent) in enumerate([
            (p1n, p1w, s1, P['p1']),
            (p2n, p2w, s2, P['p2']),
        ], 1):
            winner_glow = (pnum == 1 and s1 > s2) or (pnum == 2 and s2 > s1)
            col_bg      = P['p1_dim'] if (pnum == 1 and winner_glow) else \
                          P['p2_dim'] if (pnum == 2 and winner_glow) else P['panel']

            col = tk.Frame(cols, bg=col_bg)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                     padx=(0, 6) if pnum == 1 else (6, 0))

            tk.Label(col, text=name.upper(),
                     font=("Helvetica", 14, "bold"),
                     bg=col_bg, fg=accent).pack(pady=(14, 2))

            tk.Label(col,
                     text=f"{score}  word{'s' if score != 1 else ''}",
                     font=("Helvetica", 26, "bold"),
                     bg=col_bg, fg=accent).pack(pady=(4, 8))

            if winner_glow:
                tk.Label(col, text="★  WINNER  ★",
                         font=("Helvetica", 11, "bold"),
                         bg=col_bg, fg=P['gold']).pack(pady=(0, 8))

            tk.Frame(col, bg=P['sep'], height=1).pack(fill=tk.X, padx=14, pady=(0, 8))

            wlist = tk.Frame(col, bg=col_bg)
            wlist.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
            for i, w in enumerate(words, 1):
                row = tk.Frame(wlist, bg=P['card'], pady=4)
                row.pack(fill=tk.X, pady=1)
                tk.Frame(row, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
                tk.Label(row, text=f"{i:02d}",
                         font=("Courier", 10), bg=P['card'], fg=P['dim'],
                         width=3).pack(side=tk.LEFT, padx=(6, 4))
                tk.Label(row, text=w.upper(),
                         font=("Courier", 13, "bold"),
                         bg=P['card'], fg=P['text']).pack(side=tk.LEFT)
                tk.Label(row, text=f"+{len(w)}",
                         font=("Helvetica", 8),
                         bg=P['card'], fg=accent).pack(side=tk.RIGHT, padx=6)

        ft = tk.Frame(self._stage, bg=P['bg'])
        ft.pack(fill=tk.X, pady=10)
        tk.Button(ft, text="Play Again  (R)",
                  command=self._new_game,
                  bg=P['green'], fg='#000000',
                  font=("Helvetica", 12, "bold"),
                  relief=tk.FLAT, padx=24, pady=8, cursor="hand2",
                  activebackground=P['green'], activeforeground='#000').pack()

        filepath = self.logger.save()
        msg = (f"Game over — P1: {s1} words | P2: {s2} words"
               + (f"  — saved to {os.path.basename(filepath)}" if filepath else ""))
        self._status(msg + f"  Restarting in {AUTO_RESTART_DELAY_S} seconds...")
        self._auto_restart_token += 1
        token = self._auto_restart_token
        self.root.after(AUTO_RESTART_DELAY_S * 1000,
                        lambda t=token: self._maybe_auto_restart(t))

    # ── Scan-flow / turn management ───────────────────────────────────────────

    def _register_player(self, player_num, player_id):
        """Record a scanned-in player, then advance the scan flow.

        P1 scanned  → go to the P2 scan screen.
        P2 scanned  → both players are in, show the welcome screen and begin.
        Also reached via the Skip buttons (player_id=None) so a solo/anonymous
        game still works.
        """
        self._login_fired = True   # block a queued auto-login from re-firing
        pname = (get_player_name(player_id) if player_id is not None
                 else f"Player {player_num}")
        self.gs[f"p{player_num}_id"]   = player_id
        self.gs[f"p{player_num}_name"] = pname

        if player_num == 1:
            self._show_scan_screen(2)
        else:
            if self.gs.get("p1_id") is not None and self.gs.get("p2_id") is not None:
                self._show_welcome_screen()
            else:
                self._begin_rotation()

    def _begin_rotation(self):
        """Both players are logged in — start the alternating turns."""
        self.gs["turn_no"]      = 0
        self.gs["p1_words"]     = []
        self.gs["p2_words"]     = []
        self.gs["found_words"]  = set()
        self._next_turn()

    def _next_turn(self):
        """Advance to the next turn, or finish the game after the last one."""
        self.gs["turn_no"] += 1
        turn_no = self.gs["turn_no"]
        if turn_no > TOTAL_TURNS:
            self._show_results_screen()
            return

        player_num       = 1 if turn_no % 2 == 1 else 2
        player_round     = (turn_no + 1) // 2          # 1..TURNS_PER_PLAYER
        self.gs["active_player"] = player_num
        self._show_handover(player_num, player_round)

    def _show_handover(self, player_num, player_round):
        """Brief 'get ready' screen so kids can swap places before the clock
        starts. No timer, no submissions during this phase."""
        self._clear_stage()
        self.gs["phase"] = PH_HANDOVER
        accent = P['p1'] if player_num == 1 else P['p2']
        pname  = self.gs.get(f"p{player_num}_name") or f"Player {player_num}"

        self._lbl_phase.config(text=f"PLAYER {player_num}  —  GET READY", fg=accent)
        self._lbl_timer.config(text="")
        self._update_score_banner()

        mid = tk.Frame(self._stage, bg=P['bg'])
        mid.pack(expand=True)
        badge = tk.Frame(mid, bg=accent, width=110, height=110)
        badge.pack(pady=(0, 16))
        badge.pack_propagate(False)
        tk.Label(badge, text=str(player_num), font=("Helvetica", 56, "bold"),
                 bg=accent, fg='#0d0d0d').pack(expand=True)
        tk.Label(mid, text=f"{pname.upper()}  —  YOUR TURN",
                 font=("Helvetica", 26, "bold"),
                 bg=P['bg'], fg=accent).pack(pady=(0, 4))
        tk.Label(mid, text=f"Turn {player_round} of {TURNS_PER_PLAYER}"
                          f"     ·     {TURN_SECONDS} seconds",
                 font=("Helvetica", 13), bg=P['bg'], fg=P['dim']).pack(pady=(0, 18))
        self._lbl_countdown = tk.Label(mid, text=str(HANDOVER_SECONDS),
                                       font=("Helvetica", 40, "bold"),
                                       bg=P['bg'], fg=P['gold'])
        self._lbl_countdown.pack()

        # Robot: announce whose turn it is (short, name-personalised).
        if self.gs["turn_no"] == 1:
            say = (f"{pname}, you go first. Spell words from the tiles and "
                   f"press the buzzer for each one. Ready? Go!")
        else:
            say = f"{pname}, your turn! Go!"
        self.reachy.send("round_start", player=pname, say=say)
        print(f"[GAME] Handover → turn {self.gs['turn_no']}/{TOTAL_TURNS}: "
              f"P{player_num} ({pname}), round {player_round}")

        self._handover_token = getattr(self, "_handover_token", 0) + 1
        self._wait_for_speech_then(
            lambda: self._tick_handover(HANDOVER_SECONDS, player_num, self._handover_token),
            fallback_delay_ms=300)

    def _tick_handover(self, remaining, player_num, token):
        if token != getattr(self, "_handover_token", 0):
            return   # a new game / handover superseded this one
        if self.gs.get("phase") != PH_HANDOVER:
            return
        lbl = getattr(self, "_lbl_countdown", None)
        if remaining <= 0:
            self._start_turn(player_num)
            return
        if lbl and lbl.winfo_exists():
            lbl.config(text=str(remaining))
        self.root.after(1000,
            lambda: self._tick_handover(remaining - 1, player_num, token))

    def _show_welcome_screen(self):
        self._clear_stage()
        self.gs["phase"] = PH_WELCOME
        p1n = self.gs.get("p1_name") or "Player 1"
        p2n = self.gs.get("p2_name") or "Player 2"

        self._lbl_phase.config(text="WELCOME", fg=P['gold'])
        self._lbl_timer.config(text="")
        self._update_score_banner()

        mid = tk.Frame(self._stage, bg=P['bg'])
        mid.pack(expand=True)
        tk.Label(mid, text=f"Hi {p1n} and {p2n}!",
                 font=("Helvetica", 30, "bold"),
                 bg=P['bg'], fg=P['gold']).pack(pady=(0, 12))
        tk.Label(mid,
                 text=("Great, I got both of your player cards. "
                       "You'll take turns building words from the shared tiles. "
                       f"Each turn is {TURN_SECONDS} seconds. "
                       "When your word is ready, press the buzzer and I will read it. "
                       "No word can be used twice. Let's start!"),
                 font=("Helvetica", 14),
                 bg=P['bg'], fg=P['text'], justify=tk.CENTER, wraplength=520).pack(pady=(0, 20))

        tk.Label(mid, text="Put the shared tiles on the table and get ready.",
                 font=("Helvetica", 12, "bold"),
                 bg=P['bg'], fg=P['green']).pack(pady=(0, 18))
        tk.Label(mid, text="The timer starts when I'm done speaking.",
                 font=("Helvetica", 11),
                 bg=P['bg'], fg=P['dim']).pack()

        # Sent as COMPONENTS, not one string. The old single line opened with
        # "Hi {p1n} and {p2n}!" — a name PAIR, which is N^2 combinations and so
        # could never be pre-cached: every new pairing of campers paid a live
        # Piper synth (~4-5s of dead air) right after login. Split this way,
        # each part is its own cache entry: two per-name hellos, then fixed
        # blocks shared by every pairing. They queue back-to-back and play as
        # one continuous welcome, so nothing changes for the kids.
        #
        # _SPEED_WELCOME_RULES is mirrored byte-for-byte in
        # reachy_feedback_server.py — change one, change both.
        self.reachy.send("player_welcome",
                         lines=[f"Hi {p1n}!",
                                f"Hi {p2n}!",
                                "I got both cards!",
                                _SPEED_WELCOME_RULES,
                                f"{p1n}, you go first! Spell words from the "
                                f"tiles and press the buzzer for each one.",
                                "Let's go!"])
        self._wait_for_speech_then_start()

    def _wait_for_speech_then(self, callback, fallback_delay_ms=None):
        """Run callback after the robot is done speaking, or fallback."""
        if fallback_delay_ms is None:
            fallback_delay_ms = WELCOME_DELAY_MS
        busy = self.reachy.get_busy()
        if busy is None:
            self.root.after(min(fallback_delay_ms, 1200), callback)
            return
        if busy:
            self.root.after(300, lambda: self._wait_for_speech_then(callback, fallback_delay_ms))
            return
        callback()

    def _wait_for_speech_then_start(self):
        self._wait_for_speech_then(self._begin_rotation)

    def _start_turn(self, player_num):
        """Begin the active clock for player_num using the SHARED pool."""
        pool = self.gs["pool"]
        # active_words aliases the player's cumulative list, so accepted words
        # append straight into p{n}_words and persist across their turns.
        self.gs["active_words"] = self.gs[f"p{player_num}_words"]
        self.gs["turn_baseline"] = len(self.gs["active_words"])
        self.gs["active_pool"]  = pool          # kept for legacy references
        self.gs["turn_start"]   = time.time()
        self._last_activity     = time.time()   # fresh watchdog window
        self.gs["last_subs"]    = {}
        self.gs["feedback"]     = None
        self.gs["detected"]     = ""
        self.gs["time_up_fired"] = False
        self.gs["time_marks_left"] = [m for m in TURN_TIME_MARKS if m < TURN_SECONDS]
        self._stable_read       = ""
        self._stable_frames     = 0
        self._hands_wait_active = False   # cancel any wait from the last turn

        pname = self.gs.get(f"p{player_num}_name") or f"Player {player_num}"
        print(f"[GAME] Turn {self.gs['turn_no']}/{TOTAL_TURNS} start — "
              f"P{player_num} ({pname}) — {TURN_SECONDS}s — shared pool "
              f"{''.join(pool)}")

        self._show_turn_screen(player_num)

    def _end_turn(self, player_num):
        """Turn clock hit zero: log the words gained this turn, then rotate."""
        baseline  = self.gs.get("turn_baseline", 0)
        all_words = list(self.gs[f"p{player_num}_words"])
        new_words = all_words[baseline:]                 # gained THIS turn
        dur       = time.time() - (self.gs["turn_start"] or time.time())
        pid       = self.gs[f"p{player_num}_id"]

        self.logger.record_turn(player_num, pid, self.gs["pool"],
                                new_words, dur, self.gs["turn_no"])
        self.reachy.send("round_end", player=player_num, score=len(all_words))
        self.root.after(1200, self._next_turn)

    # ── Inactivity end ────────────────────────────────────────────────────────

    def _end_by_inactivity(self):
        """Nobody is around — wrap up the game and say goodbye.

        If a turn was in progress its words are kept and recorded, then we
        jump straight to the results screen (which fires the goodbye).
        """
        phase = self.gs.get("phase")
        if phase not in (PH_P1_TURN, PH_P2_TURN, PH_HANDOVER):
            return
        print(f"[GAME] Inactivity timeout ({INACTIVITY_TIMEOUT_S}s) — ending game.")
        self.gs["ended_by_inactivity"] = True
        self._status("Nobody's playing — ending the game. See you next time!")

        if phase in (PH_P1_TURN, PH_P2_TURN):
            pnum      = 1 if phase == PH_P1_TURN else 2
            baseline  = self.gs.get("turn_baseline", 0)
            new_words = list(self.gs[f"p{pnum}_words"])[baseline:]
            dur       = time.time() - (self.gs["turn_start"] or time.time())
            pid       = self.gs[f"p{pnum}_id"]
            self.logger.record_turn(pnum, pid, self.gs["pool"],
                                    new_words, dur, self.gs["turn_no"])

        self._show_results_screen()

    # ── Word submission ───────────────────────────────────────────────────────

    def _submit_gate(self):
        """Buzzer/SPACE during a turn lands here instead of _submit_word().

        Two paths:

        FAST — the reading has already been identical for
        HANDS_FAST_STABLE_FRAMES frames at the moment of the press, so the
        hands were clear before the buzzer was hit. Judge immediately; no
        delay. This is the common case for a kid who lays the word out, sits
        back, then presses.

        SLOW — anything else waits HANDS_DELAY_S first, because a hand still
        over the tiles reads as a stable-but-partial word and can't be
        trusted. After the delay, the first stable reading is judged (up to
        HANDS_WAIT_MAX_S more); on timeout the current reading is judged
        as-is so a press always gets an answer.
        """
        if self._hands_wait_active:
            return   # already reading a press — ignore re-presses

        # ── FAST PATH ────────────────────────────────────────────────────
        if (self.gs.get("detected", "")
                and self._stable_frames >= HANDS_FAST_STABLE_FRAMES):
            print(f"[GAME] buzzer pressed — reading already stable for "
                  f"{self._stable_frames} frames; judging now")
            self._submit_word()
            return

        # ── SLOW PATH ────────────────────────────────────────────────────
        self._hands_wait_active   = True
        self._hands_wait_ready_at = time.time() + HANDS_DELAY_S
        self._hands_wait_deadline = self._hands_wait_ready_at + HANDS_WAIT_MAX_S
        self._set_feedback("Got it! Hands away — reading your word…", ok=True)
        print("[GAME] buzzer pressed — reading tiles in "
              f"{HANDS_DELAY_S:.0f}s (hands-away delay)")
        self.root.after(HANDS_POLL_MS, self._poll_hands_wait)

    def _poll_hands_wait(self):
        if not self._hands_wait_active:
            return
        # Turn may have ended (time up / results) while we waited.
        if self.gs.get("phase") not in (PH_P1_TURN, PH_P2_TURN):
            self._hands_wait_active = False
            return
        now = time.time()
        if now < self._hands_wait_ready_at:
            self.root.after(HANDS_POLL_MS, self._poll_hands_wait)
            return   # still inside the fixed post-press delay
        detected  = self.gs.get("detected", "")
        stable_ok = bool(detected) and self._stable_frames >= HANDS_STABLE_FRAMES
        timed_out = now >= self._hands_wait_deadline
        if stable_ok or timed_out:
            self._hands_wait_active = False
            if timed_out and not stable_ok:
                print("[GAME] hands-wait timed out — judging current reading as-is")
            if self.gs.get("detected", ""):
                self._submit_word()
            else:
                self._set_feedback("I don't see any tiles!", ok=False)
            return
        self.root.after(HANDS_POLL_MS, self._poll_hands_wait)

    def _submit_word(self):
        detected = self.gs.get("detected", "").upper()
        if not detected:
            return

        # Minimum length check
        if len(detected) < MIN_WORD_LENGTH:
            self._set_feedback(f"Need at least {MIN_WORD_LENGTH} letters!", ok=False)
            self.reachy.send("word_invalid", reason="too_short", word=detected)
            return

        # Maximum length check. Only MAX_WORD_LENGTH tiles fit in the play
        # area, so a longer reading is a MISREAD (ghost tile / stray marker),
        # not a real attempt. Distinct reason= so the logs separate detection
        # faults from genuine wrong answers.
        if len(detected) > MAX_WORD_LENGTH:
            self._set_feedback(f"I see too many letters — I read "
                               f"'{detected}'. Clear the extra tiles!", ok=False)
            self.reachy.send("word_invalid", reason="too_long", word=detected)
            print(f"[GAME] MISREAD: {len(detected)} letters ('{detected}') but "
                  f"only {MAX_WORD_LENGTH} fit the play area — likely a ghost tile")
            return

        # Cooldown — prevents accidental double-submits
        last_t = self.gs["last_subs"].get(detected, 0)
        if time.time() - last_t < SUBMIT_COOLDOWN_S:
            return

        self.gs["last_subs"][detected] = time.time()

        # Already found — by EITHER player, this game (global no-repeat).
        if detected in self.gs["found_words"]:
            mine = detected.lower() in [w.lower() for w in self.gs["active_words"]]
            msg  = (f"{detected} — you already found that!" if mine
                    else f"{detected} — already taken!")
            self._set_feedback(msg, ok=False)
            self.reachy.send("word_invalid", reason="duplicate", word=detected)
            return

        # Not in dictionary
        if detected not in WORDS:
            self._set_feedback(f"'{detected}' — not a valid word", ok=False)
            self.reachy.send("word_invalid", reason="not_a_word", word=detected)
            return

        # Uses letters not in the shared pool
        pool = self.gs["pool"]
        pc   = Counter(c.upper() for c in pool)
        wc   = Counter(detected)
        if not all(pc[c] >= n for c, n in wc.items()):
            bad = [c for c, n in wc.items() if pc[c] < n]
            self._set_feedback(f"Can't use  {' '.join(bad)}  from the tiles!", ok=False)
            self.reachy.send("word_invalid", reason="bad_letters", word=detected)
            return

        # ✓ Accepted — add to this player's list AND the global lock.
        self.gs["active_words"].append(detected.lower())
        self.gs["found_words"].add(detected)
        self._set_feedback(f"✓  {detected}  +1", ok=True)
        self.reachy.send("word_found", word=detected,
                         total=len(self.gs["active_words"]))
        self._refresh_words_panel()
        self._status(
            f"Nice! '{detected}' added — "
            f"{len(self.gs['active_words'])} word{'s' if len(self.gs['active_words'])!=1 else ''} so far")
        print(f"[GAME] Accepted: {detected}  (total: {len(self.gs['active_words'])})")

    def _set_feedback(self, msg, ok=True):
        fb = getattr(self, '_lbl_feedback', None)
        if fb and fb.winfo_exists():
            fb.config(text=msg, fg=P['green'] if ok else P['red'])
        self.gs["feedback"]      = msg
        self.gs["feedback_ok"]   = ok
        self.gs["feedback_time"] = time.time()

    # ── 100 ms tick ───────────────────────────────────────────────────────────

    def _tick(self):
        try:
            now   = time.time()
            phase = self.gs.get("phase", "")

            if phase in (PH_P1_TURN, PH_P2_TURN):
                elapsed   = now - (self.gs["turn_start"] or now)
                remaining = max(0, TURN_SECONDS - int(elapsed))
                mm, ss    = divmod(remaining, 60)
                self._lbl_timer.config(
                    text=f"{mm:02d}:{ss:02d}",
                    fg=P['red'] if remaining <= 10 else P['gold'])

                # Announce the time left at each configured mark (e.g. 30s,
                # 10s). These are best-effort: the server treats time_warning
                # as ambient and DROPS it if the robot is already speaking, so
                # a kid's word feedback wins the collision instead of being
                # talked over. The mark is popped either way — a skipped
                # announcement is skipped for good, not deferred. The on-screen
                # clock is unaffected and keeps counting every tick.
                marks = self.gs.get("time_marks_left", [])
                while marks and remaining <= marks[0]:
                    mark = marks.pop(0)
                    self.reachy.send("time_warning", remaining=mark,
                                     say=_time_left_phrase(mark))

                # Time's up — fire once
                if remaining <= 0 and not self.gs.get("time_up_fired"):
                    self.gs["time_up_fired"] = True
                    pnum = 1 if phase == PH_P1_TURN else 2
                    self.reachy.send("time_up", player=pnum)
                    fb = getattr(self, '_lbl_feedback', None)
                    if fb and fb.winfo_exists():
                        fb.config(text="TIME'S UP!", fg=P['gold'])
                    self._lbl_timer.config(text="00:00", fg=P['red'])
                    # Block submissions during the brief pause before rotating.
                    self.gs["phase"] = PH_HANDOVER
                    self.root.after(900, lambda p=pnum: self._end_turn(p))

            # Feedback timeout
            if (self.gs.get("feedback")
                    and now - self.gs.get("feedback_time", 0) > FEEDBACK_DURATION):
                self.gs["feedback"] = None
                fb = getattr(self, '_lbl_feedback', None)
                if fb and fb.winfo_exists():
                    fb.config(text="")

            # ── Inactivity watchdog ─────────────────────────────────────────
            # Active once the rotation has started (turns + handovers). Ends
            # the game only when BOTH signals are quiet: the robot's camera
            # hasn't seen a face AND the game camera has seen no tiles/cards/
            # buzzer for INACTIVITY_TIMEOUT_S.
            if phase in (PH_P1_TURN, PH_P2_TURN, PH_HANDOVER):
                quiet_local = now - self._last_activity > INACTIVITY_TIMEOUT_S
                quiet_robot = not self._robot_sees_face_within(INACTIVITY_TIMEOUT_S)
                if quiet_local and quiet_robot:
                    self._end_by_inactivity()

        except Exception:
            pass

        self.root.after(100, self._tick)

    # ── Buzzer polling ────────────────────────────────────────────────────────

    def _poll_buzzer(self):
        if _buzzer_event.is_set():
            _buzzer_event.clear()
            self._on_space()
        self.root.after(50, self._poll_buzzer)

    # ── Space / buzzer handler ────────────────────────────────────────────────

    def _on_space(self):
        self._last_activity = time.time()
        phase = self.gs.get("phase")
        if phase == PH_READY:
            self._show_scan_screen(1)
        elif phase == PH_SCAN_P1:
            ids = list(self._last_scanned_ids)
            if ids:
                self._register_player(1, ids[0])
            else:
                self._status("No card detected yet. Show your ID card to log in.")
        elif phase == PH_SCAN_P2:
            ids = list(self._last_scanned_ids)
            # On the P2 scan, ignore P1's still-visible card.
            used = self.gs.get("p1_id")
            ids  = [i for i in ids if i != used]
            if ids:
                self._register_player(2, ids[0])
            else:
                self._status("No card detected yet. Show your ID card to log in.")
        elif phase in (PH_P1_TURN, PH_P2_TURN):
            self._submit_gate()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not _TK_OK:
        print("ERROR: tkinter is required.")
        return

    # Enumerate cameras ONCE up front and warm camera_select's cache, so
    # neither the recorder nor the detection camera loop pays pygrabber's
    # multi-second DirectShow enumeration cost again (both reuse the cache).
    try:
        import time as _t
        from camera_select import list_cameras
        _t0 = _t.time()
        _cams = list_cameras()
        print(f"[CAMERA] enumerated {len(_cams)} cameras in "
              f"{_t.time() - _t0:.2f}s: {_cams}")
    except Exception as _e:
        print(f"[CAMERA] camera enumeration failed: {_e}")

    # Start the external-camera recording for the whole run.
    recorder = None
    if SessionRecorder is not None:
        try:
            # Pin recording to the correct camera by NAME (not a fragile index).
            # detection uses the C920; recording uses the j5. Pass the index
            # DIRECTLY to the constructor — the env var is read at import time,
            # too early to influence this, so cam_index=... is the reliable path.
            from camera_select import recording_index
            rec_idx = recording_index()
            recorder = SessionRecorder(cam_index=rec_idx).start()
            atexit.register(recorder.stop)   # belt-and-suspenders finalize
        except Exception as e:
            print(f"[REC] could not start recorder: {e}")

    # On Windows, closing the console/terminal window (or a logoff/shutdown)
    # kills the process directly — it does NOT raise KeyboardInterrupt and
    # does NOT run atexit handlers or the `finally` block below in time,
    # so the recorder never gets its graceful "q"/SIGINT stop. That leaves
    # the in-progress ffmpeg segment unfinalized, i.e. no usable video for
    # that whole session. Catching the console close event here gives us a
    # last chance to stop the recorder cleanly before Windows tears the
    # process down.
    if platform.system() == "Windows" and recorder is not None:
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
                    recorder.stop()
                return False   # let Windows continue its own shutdown handling

            # Keep a reference alive at module scope so it isn't garbage
            # collected — a GC'd ctypes callback would crash on invocation.
            global _console_ctrl_handler_ref
            _console_ctrl_handler_ref = HandlerRoutine(_on_console_ctrl)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(
                _console_ctrl_handler_ref, True)
        except Exception as e:
            print(f"[REC] could not install console-close handler: {e}")

    root = tk.Tk()
    SpeedGameGUI(root)

    # Let Python process Ctrl-C promptly even while tk's mainloop is blocking.
    def _pump():
        root.after(200, _pump)
    root.after(200, _pump)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[REC] Ctrl-C received — finalizing recording…")
    finally:
        if recorder:
            recorder.stop()   # idempotent; safe alongside atexit


if __name__ == "__main__":
    main()