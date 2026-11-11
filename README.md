# Scrambled Words

A physical word-building game for kids, played with printed letter cards and a Reachy Mini robot. Players hold up ArUco-marked letter tiles in front of a camera, the game reads them in real time, and the robot cheers, hints and explains the rules out loud.

It was built for a children's summer STEM camp (ages 6 to 12) and runs fully offline: a laptop and the robot only need to share a local Wi-Fi network or hotspot.

## Game modes

- **Solo scramble** (`game.py`): unscramble a word by placing the right letter cards in the play area. Includes hints, categories and levels.
- **Speed builder** (`speed_game.py`): two players race to build words, each on their own side of the play area.

Players log in by showing an ID card, so the robot can greet them and the session log can track each player.

## How it works

| Part | Runs on | What it does |
|---|---|---|
| `detection.py` | Laptop | OpenCV and ArUco marker detection shared by both games |
| `game.py`, `speed_game.py` | Laptop | Camera loop, game logic and the on-screen GUI (tkinter) |
| `reachy_feedback_server.py` | Robot | Small HTTP server that turns game events into speech and gestures |
| `phrases.json` | Robot | Pre-written spoken lines the robot picks from |
| `generate_phrases.py` | Laptop | One-time script that uses a local LLM to write `phrases.json` |
| `game_logger.py` | Laptop | Writes one Excel file per session (summary sheet and per-word sheet) |
| `session_recorder.py` | Laptop | Optional session video recording |
| `player_names.py` | Both | Maps player ID cards to display names |

The laptop sends short events (`card_detected`, `correct`, `wrong_letter`, `level_complete`, and so on) to the robot over HTTP. Each event type has a cooldown, and the robot's speech queue caps its backlog, so the robot never rambles behind the game.

### Marker scheme

Three separate ArUco dictionaries keep every marker type distinct, so IDs can never collide:

| Marker | Dictionary | IDs |
|---|---|---|
| Play-area corners | `DICT_4X4_250` | 10 to 13 |
| Letter and word cards | `DICT_ARUCO_ORIGINAL` | 0 to 199 |
| Player ID cards | `DICT_6X6_1000` | 200 to 999 |

### Where the LLM fits

The robot's dialogue is generated ahead of time with **Gemma 3 4B** running locally through **Ollama** (`generate_phrases.py`), which writes about 10 kid-friendly variations of every line to `phrases.json`. During play the robot just picks a line at random, so there is no model latency and no internet dependency. If `phrases.json` is missing, the server falls back to a small built-in set.

## Setup

1. Install the laptop dependencies: `pip install -r requirements.txt`
2. Copy `player_names.example.json` to `player_names.json` and add your own players. `player_names.json` is git-ignored so real names are never committed.
3. Print the ArUco letter cards, player ID cards and the four play-area corner markers (`custom_markers/` holds the letter templates the detector uses).
4. Start the feedback server on the robot, then run a game on the laptop.

Full step-by-step instructions (copying files to the robot, voice settings, controls) are in [SETUP.md](SETUP.md).

## Controls

| Key | `game.py` | `speed_game.py` |
|---|---|---|
| `SPACE` or buzzer | Ready / check word | Submit word |
| `H` | Hint | |
| `C` | Change category | |
| `R` | New game | New game |
| `Q` | Quit | Quit |

## Built with

Python, OpenCV (ArUco), tkinter, Reachy Mini SDK, Piper text-to-speech, Ollama with Gemma 3 4B, openpyxl.

## Privacy

This repository contains no player data. Player names, session logs and recordings stay on the machine that runs the game and are excluded by `.gitignore`.
