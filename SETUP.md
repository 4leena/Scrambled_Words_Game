# Word Scramble + Reachy Mini Feedback

## What's in this folder

| File                          | Runs on | Purpose |
|--------------------------------|---------|---------|
| `game.py`                     | Mac     | Solo scramble mode — camera, GUI, game logic |
| `speed_game.py`                | Mac     | Two-player speed builder mode |
| `detection.py`                 | Mac     | OpenCV / ArUco marker detection shared by both games |
| `game_logger.py`                | Mac     | Writes `.xlsx` session logs |
| `player_names.py`               | Both    | Looks up camper names from `player_names.json` |
| `player_names.json`             | Both    | **Edit this** — maps ArUco player-card IDs to camper names |
| `reachy_feedback_server.py`      | Robot   | HTTP server on Reachy — turns game events into speech + gestures |
| `phrases.json`                  | Robot   | Pre-written spoken phrase variations (see "How the LLM fits in" below) |
| `generate_phrases.py`           | Mac     | One-time script — uses Ollama to regenerate `phrases.json` |
| `requirements.txt`              | —       | Python dependencies (see file for Mac vs robot split) |
| `A.jpg` ... `Z.jpg`                | Mac     | Letter-tile template images used by `detection.py` |
| `BOUNDARY_12.png`, `BOUNDARY_13.png` | Mac (print) | Printable boundary corner markers |

## Architecture

- The Mac runs `game.py` / `speed_game.py` because the external camera is connected there.
- Reachy Mini runs `reachy_feedback_server.py` because robot movement and speech happen on the robot.
- They communicate over the local network at `http://reachy-mini.local:8765`.
- No internet is required at camp. Both devices only need to be on the same Wi-Fi/hotspot.

## How the LLM fits in (Ollama, not a cloud API)

There is no live LLM call during gameplay -- this is intentional, to avoid any
latency or internet dependency while kids are playing.

1. Before camp (one-time, on your Mac): run `generate_phrases.py`. It calls a
   locally running Ollama server (`ollama serve`) using the Gemma 3 4B model
   to generate roughly 10 varied, kid-friendly phrasings for every spoken
   event (intro, correct answer, wrong answer, level up, etc). The results
   are saved to `phrases.json`.
2. At camp (runtime, on the robot): `reachy_feedback_server.py` just loads
   `phrases.json` and picks a random variation for whichever event fires
   (`random.choice`, avoiding immediate repeats). This is instant -- no model
   inference happens live.
3. If `phrases.json` is missing or corrupt, the server falls back to a small
   built-in set of default phrases (`_PHRASES_FALLBACK` in
   `reachy_feedback_server.py`), so the game still runs even on a fresh setup.

To regenerate phrases (e.g. after editing the wording):
```bash
pip install ollama
ollama pull gemma3:4b
ollama serve                  # leave running in its own terminal
python3 generate_phrases.py   # on your Mac
scp phrases.json pollen@10.42.0.72:~/Scramble_Words/
```

## Player names (no GUI editing needed)

Copy `player_names.example.json` to `player_names.json` (it is git-ignored so real names never get committed), then edit it -- it's a plain JSON map of ArUco player-card
ID to camper name:
```json
{ "100": "Maya", "101": "Leo" }
```
Any ID not listed falls back to a generic "Player <id>" label. This file must
exist on both the Mac (so on-screen labels show real names) and the robot
(so Reachy's spoken welcome uses real names) -- keep both copies in sync.

## How the game-to-robot feedback flow works

1. Camper(s) scan their ArUco ID card(s). Reachy first plays a generic
   `login_prompt` ("show me your ID card") before any card is seen.
2. Once an ID is detected, the game sends `player_login` with the camper's
   real name (looked up via `player_names.py`) -- Reachy gives a personalized
   welcome.
3. A few seconds later, the game sends `instructions` with the mode
   (`scramble` or `speed`) -- Reachy explains that specific game's rules in
   kid-friendly language.
4. During play, `game.py`/`speed_game.py` send small, specific events
   (`card_detected`, `correct`, `wrong_letter`, `level_complete`,
   `word_found`, `time_tick`, etc) over HTTP POST to
   `reachy_feedback_server.py`, which queues the matching spoken line and
   triggers a robot gesture (`excited`, `curious`, `wiggle`, `sad`, etc).
5. Throttling, so Reachy doesn't ramble: each event type has its own
   client-side cooldown (e.g. `observing` every 12s, `wrong_letter` every 8s)
   so the camera loop can't spam events. On the robot side, the speech queue
   itself enforces a minimum gap between spoken lines (`MIN_LINE_GAP_S`) and
   caps how many lines can queue up (`MAX_QUEUE_DEPTH`), so a quick burst of
   game events never turns into a long stale backlog of speech playing out
   after the moment has passed.

## Logging

`game_logger.py` writes one `.xlsx` file per session with two sheets:
"Game Summary" (one row per session, including player IDs) and "Word Detail"
(one row per word attempt). `speed_game.py` uses its own equivalent logger
for two-player sessions. Logs are written locally on the Mac.

## 1. Copy this folder to Reachy

From your laptop terminal, inside the folder that contains this project:

```bash
scp game.py speed_game.py detection.py game_logger.py player_names.py \
    player_names.json reachy_feedback_server.py phrases.json \
    pollen@10.42.0.72:~/Scramble_Words/
```

If the hostname doesn't resolve, use the robot's IP directly -- `10.42.0.72`
is this project's known robot IP; substitute `reachy-mini.local` if your
network resolves it.

## 2. Terminal 1: run the Reachy feedback server ON REACHY

```bash
ssh pollen@10.42.0.72
source /venvs/apps_venv/bin/activate
cd ~/Scramble_Words
PIPER_MODEL=/home/pollen/en_US-hfc_female-medium.onnx ALSA_DEVICE=plughw:0,0 python3 reachy_feedback_server.py
```

Leave this terminal open. Swap `PIPER_MODEL` to a different voice `.onnx`
file to A/B test -- no code change needed.

## 3. Terminal 2: run a game ON YOUR LAPTOP

```bash
python3 game.py          # solo scramble mode
# or
python3 speed_game.py    # two-player speed mode
```

`game.py` controls:
- `SPACE` / buzzer = ready / check word
- `H` = hint
- `R` = new game
- `C` = change category
- `Q` = quit

`speed_game.py` controls:
- `SPACE` / buzzer = submit word
- `R` = new game
- `Q` = quit

## If Reachy does not react

Test the server from your laptop:

```bash
curl http://10.42.0.72:8765/health
```

You should see:

```json
{"ok": true}
```

If not, check:

1. Reachy and laptop are on the same Wi-Fi/hotspot.
2. `reachy_feedback_server.py` is still running on Reachy (check the terminal
   for `[INFO] Tutor API at http://0.0.0.0:8765`).
3. Try the robot's IP directly if `reachy-mini.local` doesn't resolve.
4. If using a different IP, update `REACHY_FEEDBACK_URL` near the top of
   `game.py` and `speed_game.py`:

```python
REACHY_FEEDBACK_URL = "http://<REACHY_IP>:8765/event"
```