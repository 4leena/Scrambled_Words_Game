"""
game_logger.py
==============
Logs each game session to a single .xlsx file with two sheets:
  - "Game Summary"  : one row for the whole session (includes player IDs)
  - "Word Detail"   : one row per word attempt (includes player IDs)

HOW TO CONNECT TO game.py
--------------------------
1. Import (already done):
       from game_logger import SessionLogger

2. Create logger after player scan, passing detected player IDs:
       logger = SessionLogger(selected_category, state["player_ids"])

3. After check_word() in the SPACE/B handler (unchanged):
       if result["result"] in ("correct", "wrong", "failed"):
           status = "incorrect" if result["result"] == "wrong" else result["result"]
           attempts_used = state["attempts"][state["current_idx"] - 1]
           logger.record_puzzle(result["target"], status, attempts_used)
           if result.get("finished"):
               logger.save(finished=True)

4. Q key (before break):
       logger.save(finished=False)

5. R/C key (before state = new_state(...)):
       logger.save(finished=False)
       logger = SessionLogger(selected_category, state["player_ids"])
"""

import json
import os
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

# Resolve ArUco player IDs to real camper names via player_names.json
# (same lookup game.py uses). Falls back to "Player <id>" if the module
# or the JSON file is missing, so logging never breaks.
try:
    from player_names import get_player_name, get_player_age
except Exception:
    def get_player_name(pid):
        return f"Player {pid}"

    def get_player_age(pid):
        return None

# -------------------------------------------------------
# COLOURS
# -------------------------------------------------------
COL_HEADER_BG   = "2F5496"
COL_HEADER_TEXT = "FFFFFF"
COL_CORRECT     = "C6EFCE"
COL_INCORRECT   = "FFEEBA"
COL_FAILED      = "FFC7CE"
COL_SUMMARY_BG  = "D9E1F2"
COL_PLAYER_BG   = "E2EFDA"   # light green — player ID cells


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def _style(cell, bold=False, bg=None, text_color=None, center=False):
    if bold or text_color:
        cell.font = Font(bold=bold, color=text_color if text_color else "000000")
    if bg:
        cell.fill = PatternFill("solid", start_color=bg)
    if center:
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _write_headers(ws, row, headers):
    for col, text in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col, value=text)
        _style(cell, bold=True, bg=COL_HEADER_BG,
               text_color=COL_HEADER_TEXT, center=True)


def _autosize_columns(ws):
    for col in ws.columns:
        max_len = max(
            (len(str(cell.value)) for cell in col if cell.value is not None),
            default=8
        )
        ws.column_dimensions[col[0].column_letter].width = max_len + 4


# =======================================================
# SessionLogger CLASS
# =======================================================

class SessionLogger:

    def __init__(self, category: str = "unknown", player_ids: list = None):
        self.start_time = datetime.now()
        self.category   = category

        # player_ids is a list of ArUco IDs detected at session start,
        # e.g. [201, 205] or [201] if only one kid scanned.
        # Each ID is resolved to the camper's real name via
        # player_names.json ("Leo", "Zoe", ...); unregistered cards fall
        # back to "Player <id>". Both the names AND the raw IDs are logged,
        # so study data stays analysable even if a name is edited later.
        raw_ids = list(player_ids) if player_ids else []
        self.player_ids   = raw_ids
        self.player_names = [get_player_name(pid) for pid in raw_ids] or ["Unknown"]

        self.player_ages  = [get_player_age(pid) for pid in raw_ids]

        # Human-readable strings for the log
        self.players_str = ", ".join(self.player_names)          # "Leo, Zoe"
        self.ids_str     = ", ".join(str(p) for p in raw_ids) or "—"   # "201, 205"
        self.ages_str    = ", ".join(str(a) if a is not None else "?"
                                     for a in self.player_ages) or "—"  # "8, 10"

        self.puzzles = []
        self._word_start = datetime.now()

        print(f"[LOG] Session started at {self.start_time.strftime('%H:%M:%S')} "
              f"| category: {category} | players: {self.players_str}")


    def record_puzzle(self, word: str, status: str, attempts_used: int):
        word_end = datetime.now()
        self.puzzles.append({
            "word":       word.upper(),
            "status":     status,
            "attempts":   attempts_used,
            "word_start": self._word_start,
            "word_end":   word_end,
        })
        self._word_start = datetime.now()
        print(f"[LOG] Word recorded: {word.upper()} → {status} "
              f"(attempts: {attempts_used})")


    def save(self, finished: bool):

        end_time     = datetime.now()
        elapsed_s    = int((end_time - self.start_time).total_seconds())
        duration_str = f"{elapsed_s // 60}m {elapsed_s % 60}s"
        n_correct    = sum(1 for p in self.puzzles if p["status"] == "correct")

        base_dir  = os.path.dirname(os.path.abspath(__file__))
        log_dir   = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filepath  = os.path.join(log_dir, f"session_{timestamp}.xlsx")

        wb = Workbook()

        # ======================
        # SHEET 1: Game Summary
        # ======================
        ws_summary       = wb.active
        ws_summary.title = "Game Summary"

        summary_headers = [
            "Date", "Start Time", "End Time", "Duration",
            "Category", "Players", "Player IDs", "Ages",  # ← names, IDs, ages
            "Words Solved", "Out Of", "Finished?"
        ]
        _write_headers(ws_summary, row=1, headers=summary_headers)

        ws_summary.append([
            self.start_time.strftime("%Y-%m-%d"),
            self.start_time.strftime("%H:%M:%S"),
            end_time.strftime("%H:%M:%S"),
            duration_str,
            self.category,
            self.players_str,                             # ← e.g. "Leo, Zoe"
            self.ids_str,                                 # ← e.g. "201, 205"
            self.ages_str,                                # ← e.g. "8, 10"
            n_correct,
            len(self.puzzles),
            "Yes" if finished else "No",
        ])

        for col in range(1, len(summary_headers) + 1):
            cell = ws_summary.cell(row=2, column=col)
            _style(cell, bg=COL_SUMMARY_BG, center=True)

        # Highlight the Players / Player IDs / Ages cells in green
        # Columns 6-8 = "Players", "Player IDs", "Ages"
        for col in (6, 7, 8):
            _style(ws_summary.cell(row=2, column=col),
                   bg=COL_PLAYER_BG, center=True)

        # Finished? cell: green or red (column 11)
        _style(ws_summary.cell(row=2, column=11),
               bg=COL_CORRECT if finished else COL_FAILED, center=True)

        _autosize_columns(ws_summary)

        # ======================
        # SHEET 2: Word Detail
        # ======================
        ws_detail = wb.create_sheet(title="Word Detail")

        detail_headers = [
            "Date", "Session Start", "Category",
            "Players", "Player IDs", "Ages",              # ← names, IDs, ages
            "Word #", "Word",
            "Word Start", "Word End", "Word Duration",
            "Attempts Used", "Max Attempts", "Result"
        ]
        _write_headers(ws_detail, row=1, headers=detail_headers)

        for i, puzzle in enumerate(self.puzzles, start=1):
            word_elapsed = int(
                (puzzle["word_end"] - puzzle["word_start"]).total_seconds()
            )
            word_dur_str = f"{word_elapsed // 60}m {word_elapsed % 60}s"

            ws_detail.append([
                self.start_time.strftime("%Y-%m-%d"),
                self.start_time.strftime("%H:%M:%S"),
                self.category,
                self.players_str,                         # ← same for every row in this session
                self.ids_str,
                self.ages_str,
                i,
                puzzle["word"],
                puzzle["word_start"].strftime("%H:%M:%S"),
                puzzle["word_end"].strftime("%H:%M:%S"),
                word_dur_str,
                puzzle["attempts"],
                3,
                puzzle["status"],
            ])

            # Highlight Players / Player IDs / Ages cells (columns 4-6)
            for col in (4, 5, 6):
                _style(ws_detail.cell(row=i + 1, column=col),
                       bg=COL_PLAYER_BG, center=True)

            # Colour-code Result cell (column 14)
            result_cell = ws_detail.cell(row=i + 1, column=14)
            if puzzle["status"] == "correct":
                _style(result_cell, bg=COL_CORRECT, center=True)
            elif puzzle["status"] == "incorrect":
                _style(result_cell, bg=COL_INCORRECT, center=True)
            else:
                _style(result_cell, bg=COL_FAILED, center=True)

        _autosize_columns(ws_detail)

        wb.save(filepath)
        print(f"[LOG] Saved → {filepath}")

        # ======================
        # JSON LOGS (for the study)
        # ======================
        # 1. logs/session_<timestamp>.json — full machine-readable mirror of
        #    this session (overwritten on re-save so it always holds the
        #    latest data).
        # 2. logs/sessions_master.jsonl — ONE line per session, all sessions
        #    in one file. Load the whole study in one call:
        #        import pandas as pd
        #        df = pd.read_json("logs/sessions_master.jsonl", lines=True)
        #    save() can run more than once per session (timeout save + kiosk
        #    reset save), so the master is deduped by session_id — each
        #    session appears exactly once, with its most recent data.
        session_id = timestamp
        data = {
            # Discriminator so scramble and speed records can be concatenated
            # despite having different shapes (puzzles vs turns). Purely
            # additive — older logs simply lack the key.
            "game":             "scramble",
            "session_id":       session_id,
            "date":             self.start_time.strftime("%Y-%m-%d"),
            "start_time":       self.start_time.strftime("%H:%M:%S"),
            "end_time":         end_time.strftime("%H:%M:%S"),
            "duration_seconds": elapsed_s,
            "category":         self.category,
            "players": [
                {"id": pid, "name": get_player_name(pid),
                 "age": get_player_age(pid)}
                for pid in self.player_ids
            ] or [{"id": None, "name": "Unknown", "age": None}],
            "words_solved":     n_correct,
            "out_of":           len(self.puzzles),
            "finished":         bool(finished),
            "puzzles": [
                {
                    "word_number":      i,
                    "word":             p["word"],
                    "result":           p["status"],
                    "attempts_used":    p["attempts"],
                    "max_attempts":     3,
                    "word_start":       p["word_start"].strftime("%H:%M:%S"),
                    "word_end":         p["word_end"].strftime("%H:%M:%S"),
                    "duration_seconds": int((p["word_end"] -
                                             p["word_start"]).total_seconds()),
                }
                for i, p in enumerate(self.puzzles, start=1)
            ],
        }

        json_path = os.path.join(log_dir, f"session_{session_id}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[LOG] Saved → {json_path}")
        except Exception as e:
            print(f"[LOG] WARN: could not write {json_path}: {e}")

        master_path = os.path.join(log_dir, "sessions_master.jsonl")
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

        return filepath