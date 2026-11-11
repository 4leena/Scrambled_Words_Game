"""
session_recorder.py
===================
Records the WHOLE run from an external camera (a *second* camera, separate
from the ArUco detection camera) for the research study — now with AUDIO and
crash-safe segmenting.

TWO BACKENDS
------------
1. ffmpeg (preferred, macOS): captures the external camera AND a microphone in
   ONE synced pipeline via avfoundation, burns in a timestamp, and writes
   crash-safe .mp4 chunks. This is the right path for a study that analyses how
   kids react to the robot's SPEECH — video and audio share one clock so they
   can't drift apart over a long session.
2. OpenCV (fallback, video-only): used automatically if ffmpeg is missing,
   audio is turned off, or you're not on macOS. No sound.

Either way the public API is identical, so game.py's hook never changes:
    rec = SessionRecorder().start()   # begins recording immediately
    ...run the whole kiosk...
    rec.stop()                        # finalizes the file(s); safe to call twice

CRASH SAFETY
------------
Recording is chunked (default every 10 min). Every COMPLETED chunk is a normal,
double-click-playable .mp4. A graceful shutdown (terminal Ctrl-C -> SIGINT to
ffmpeg, or OpenCV writer release) finalizes the current chunk too, so the
normal end-of-day case loses nothing. A hard power-loss costs you at most the
one in-progress chunk (<= RECORD_SEGMENT_MIN minutes), never the whole run.
Set RECORD_SEGMENT_MIN=0 for a single continuous file (less crash-safe).

FIND YOUR DEVICE INDICES (do this once, on the Mac)
---------------------------------------------------
ffmpeg and OpenCV number cameras DIFFERENTLY, so list both:

    python3 session_recorder.py --list        # shows ffmpeg AV devices + OpenCV cams
    python3 session_recorder.py                # lists, then records a 5s A/V test clip

The detection camera is OpenCV index 0. Point the recorder at the OTHER camera.
For the ffmpeg backend you give it that camera's *avfoundation* video index and
a microphone's audio index (from the --list output).

CONFIG (env vars -- set before launching game.py, or edit the defaults)
    RECORD_AUDIO        1 = record sound (default 1); 0 = video only
    RECORD_FF_VIDEO     ffmpeg avfoundation VIDEO device index  (default "1")
    RECORD_FF_AUDIO     ffmpeg avfoundation AUDIO device index  (default "0")
    RECORD_CAM_INDEX    OpenCV camera index for the fallback    (default 1)
    RECORD_FPS          frames/sec                              (default 30)
    RECORD_SIZE         e.g. 1280x720; empty = camera default   (default "")
    RECORD_DIR          output folder                           (default ./recordings)
    RECORD_SEGMENT_MIN  minutes per chunk; 0 = one file         (default 10)
"""

import os
import sys
import time
import shutil
import signal
import platform
import threading
import subprocess
from datetime import datetime

import cv2   # only used by the OpenCV fallback + camera listing


# -- Defaults (overridable via env vars) --------------------------------------
_DEF_AUDIO       = os.environ.get("RECORD_AUDIO", "1") == "1"
_DEF_FF_VIDEO    = os.environ.get("RECORD_FF_VIDEO", "1")
_DEF_FF_AUDIO    = os.environ.get("RECORD_FF_AUDIO", "0")
_DEF_CAM_INDEX   = int(os.environ.get("RECORD_CAM_INDEX", "1"))
_DEF_FPS         = float(os.environ.get("RECORD_FPS", "30"))
_DEF_SIZE        = os.environ.get("RECORD_SIZE", "").strip()   # "" = camera default
_DEF_DIR         = os.environ.get(
    "RECORD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "recordings"))
_DEF_SEGMENT_MIN = int(os.environ.get("RECORD_SEGMENT_MIN", "10"))

# -- Windows (dshow) audio+video recording ------------------------------------
# On Windows, ffmpeg captures via DirectShow using DEVICE NAMES (from
# `ffmpeg -list_devices true -f dshow -i dummy`). Set these to the recording
# camera and the microphone you want in the session videos.
_DEF_WIN_VIDEO = os.environ.get("RECORD_WIN_VIDEO", "j5 WebCam JVCU100")
_DEF_WIN_AUDIO = os.environ.get("RECORD_WIN_AUDIO", "Microphone (Mic-j5 WebCam JVCU100)")

# Video encoder for the ffmpeg path. Default to libx264 (software) on all
# platforms: for a single webcam at 30 fps the "ultrafast" preset is very
# light on CPU, and it starts recording immediately. Intel QuickSync
# (h264_qsv) was the old Windows default, but its runtime init on Windows can
# stall for 20+ seconds before the first frame is written — that was the
# startup delay. To go back to GPU encoding, set RECORD_VENC=h264_qsv.
_DEF_VENC    = os.environ.get("RECORD_VENC", "libx264")
_DEF_VPRESET = os.environ.get("RECORD_VPRESET", "ultrafast")

_IS_MAC   = platform.system() == "Darwin"
_IS_WIN   = platform.system() == "Windows"

# Locate ffmpeg. Prefer PATH; otherwise fall back to the known winget install
# location so recording works even when ffmpeg isn't on PATH.
def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Allow an explicit override.
    env_path = os.environ.get("FFMPEG_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path
    # Search the winget Packages folder for ffmpeg.exe (version-agnostic).
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                        "Microsoft", "WinGet", "Packages")
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
    return None

_FFMPEG   = _find_ffmpeg()
_HAS_FF   = _FFMPEG is not None


class SessionRecorder:
    """Continuously records the external camera (with audio when possible) to
    timestamped, crash-safe .mp4 chunks."""

    def __init__(self, audio=None, ff_video=None, ff_audio=None, cam_index=None,
                 fps=None, size=None, out_dir=None, segment_minutes=None,
                 timestamp=True, fail_hard=False):
        self.audio       = _DEF_AUDIO if audio is None else bool(audio)
        self.ff_video    = _DEF_FF_VIDEO if ff_video is None else str(ff_video)
        self.ff_audio    = _DEF_FF_AUDIO if ff_audio is None else str(ff_audio)
        self.cam_index   = _DEF_CAM_INDEX if cam_index is None else int(cam_index)
        self.fps         = _DEF_FPS if fps is None else float(fps)
        self.size        = _DEF_SIZE if size is None else size
        self.out_dir     = _DEF_DIR if out_dir is None else out_dir
        self.segment_min = _DEF_SEGMENT_MIN if segment_minutes is None \
            else int(segment_minutes)
        self.timestamp   = timestamp
        # If True, refuse to run when recording can't start (missing footage may
        # be worse for the study than a delayed session).
        self.fail_hard   = fail_hard

        self.backend     = None      # "ffmpeg" | "opencv" | None
        self._proc       = None      # ffmpeg subprocess
        self._logf       = None
        self._stopped    = False

        # OpenCV-fallback state
        self._cap        = None
        self._writer     = None
        self._thread     = None
        self._stop       = threading.Event()
        self._paths      = []
        self._seg_start  = 0.0

    # -- public API -----------------------------------------------------------
    def start(self):
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[REC] Output folder: {os.path.abspath(self.out_dir)}")

        # Choose backend: ffmpeg for synced audio+video on macOS (avfoundation)
        # or Windows (dshow); otherwise fall back to OpenCV (video only).
        want_ff_mac = self.audio and _HAS_FF and _IS_MAC
        want_ff_win = self.audio and _HAS_FF and _IS_WIN
        if self.audio and not (want_ff_mac or want_ff_win):
            why = ("ffmpeg not found"
                   if not _HAS_FF else
                   "audio capture only supported on macOS/Windows here")
            print(f"[REC] Audio requested but unusable: {why}. "
                  f"Falling back to VIDEO-ONLY recording.")

        if want_ff_mac:
            self._start_ffmpeg()
        elif want_ff_win:
            self._start_ffmpeg_windows()
        else:
            self._start_opencv()
        return self

    def stop(self):
        """Finalize the recording. Idempotent -- safe from any exit path
        (Ctrl-C handler, atexit, _quit)."""
        if self._stopped:
            return
        self._stopped = True
        if self.backend == "ffmpeg":
            self._stop_ffmpeg()
        elif self.backend == "opencv":
            self._stop_opencv()
        self._report()

    # -- ffmpeg backend (audio + video, macOS avfoundation) -------------------
    def _start_ffmpeg(self):
        input_args = [
            "-f", "avfoundation",
            "-framerate", str(int(self.fps)),
        ]
        if self.size:
            input_args += ["-video_size", self.size]
        # avfoundation input spec is "VIDEO:AUDIO" (indices from --list).
        input_args += ["-i", f"{self.ff_video}:{self.ff_audio}"]

        cmd = self._build_ffmpeg_cmd(input_args)
        logpath = os.path.join(self.out_dir, "ffmpeg.log")
        try:
            self._logf = open(logpath, "a", encoding="utf-8")
            self._logf.write(f"\n=== {datetime.now()} :: {' '.join(cmd)}\n")
            self._logf.flush()
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self._logf)
            self.backend = "ffmpeg"
            self._seg_start = time.time()
            print(f"[REC] Recording (ffmpeg) -- avfoundation video:{self.ff_video} "
                  f"audio:{self.ff_audio}, {self.fps:g} fps -> {self.out_dir}"
                  + (f" (new file every {self.segment_min} min)"
                     if self.segment_min else " (single file)"))
            print(f"[REC] ffmpeg messages -> {logpath}")
            # Give ffmpeg a moment; if the device indices are wrong it exits fast.
            time.sleep(1.0)
            if self._proc.poll() is not None:
                msg = (f"[REC] ffmpeg exited immediately -- check device indices "
                       f"(video:{self.ff_video} audio:{self.ff_audio}). "
                       f"Run `python3 session_recorder.py --list`. See {logpath}.")
                self._proc = None
                self.backend = None
                if self.fail_hard:
                    raise RuntimeError(msg)
                print("=" * 70 + f"\n{msg}\nFalling back to VIDEO-ONLY.\n" + "=" * 70)
                self._start_opencv()
            else:
                self._start_watchdog()
        except Exception as e:
            self.backend = None
            if self.fail_hard:
                raise
            print(f"[REC] ffmpeg failed to start ({e}) -- falling back to video-only.")
            self._start_opencv()

    # -- ffmpeg backend (audio + video, Windows DirectShow) -------------------
    def _start_ffmpeg_windows(self, venc=None, stamp=True):
        video_name = _DEF_WIN_VIDEO
        audio_name = _DEF_WIN_AUDIO
        venc = venc or _DEF_VENC
        # dshow takes ONE -i with "video=NAME:audio=NAME".
        input_args = [
            "-f", "dshow",
            "-rtbufsize", "512M",           # cushions USB webcam bursts
            # Start writing frames fast: by default ffmpeg spends up to ~5s
            # ANALYZING a live input before it begins, which clips the start of
            # a per-game recording. A webcam's format is simple and we pass the
            # framerate explicitly, so ~0.5s of probing is plenty.
            "-analyzeduration", "500000",   # 0.5s (default is 5s)
            "-probesize", "1000000",        # 1 MB
            "-framerate", str(int(self.fps)),
        ]
        if self.size:
            input_args += ["-video_size", self.size]
        input_args += ["-i", f"video={video_name}:audio={audio_name}"]

        cmd = self._build_ffmpeg_cmd(input_args, venc=venc, stamp=stamp)
        logpath = os.path.join(self.out_dir, "ffmpeg.log")
        try:
            self._logf = open(logpath, "a", encoding="utf-8")
            self._logf.write(f"\n=== {datetime.now()} :: {' '.join(cmd)}\n")
            self._logf.flush()
            # cwd=out_dir so the relative stamp_font.ttf fontfile resolves.
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, cwd=self.out_dir,
                stdout=subprocess.DEVNULL, stderr=self._logf)
            self.backend = "ffmpeg"
            self._seg_start = time.time()
            print(f"[REC] Recording (ffmpeg/dshow, {venc}"
                  + ("" if stamp else ", no timestamp overlay")
                  + f") -- video:'{video_name}' "
                  f"audio:'{audio_name}', {self.fps:g} fps -> {self.out_dir}"
                  + (f" (new file every {self.segment_min} min)"
                     if self.segment_min else " (single file)"))
            print(f"[REC] ffmpeg messages -> {logpath}")
            # Give ffmpeg a moment; wrong device names / bad encoder exit fast.
            time.sleep(1.5)
            if self._proc.poll() is not None:
                self._proc = None
                self.backend = None
                # Self-healing ladder before giving up on audio entirely:
                # 1. hardware encoder failed -> software encoder
                if venc == "h264_qsv":
                    print("[REC] QuickSync (h264_qsv) didn't start — retrying "
                          "with software encoder (libx264, ultrafast).")
                    return self._start_ffmpeg_windows(venc="libx264",
                                                      stamp=stamp)
                # 2. software encoder failed WITH the timestamp overlay ->
                #    retry once without it (a drawtext/font problem must not
                #    cost us the audio track).
                if stamp:
                    print("[REC] ffmpeg exited with the timestamp overlay "
                          "enabled — retrying WITHOUT the burned-in "
                          "timestamp (keeping audio).")
                    return self._start_ffmpeg_windows(venc="libx264",
                                                      stamp=False)
                # 3. still dead -> video-only fallback
                msg = (f"[REC] ffmpeg exited immediately -- check device names "
                       f"(video:'{video_name}' audio:'{audio_name}'). "
                       f"List them with "
                       f"`ffmpeg -list_devices true -f dshow -i dummy`. "
                       f"See {logpath}.")
                if self.fail_hard:
                    raise RuntimeError(msg)
                print("=" * 70 + f"\n{msg}\nFalling back to VIDEO-ONLY.\n" + "=" * 70)
                self._start_opencv()
            else:
                self._start_watchdog()
        except Exception as e:
            self.backend = None
            if self.fail_hard:
                raise
            print(f"[REC] ffmpeg (dshow) failed to start ({e}) -- "
                  f"falling back to video-only.")
            self._start_opencv()

    # -- ffmpeg watchdog -------------------------------------------------------
    # The startup checks above only verify ffmpeg survived 1-1.5 seconds.
    # DirectShow problems (device busy, audio name mismatch, encoder issues)
    # often kill ffmpeg AFTER that window — leaving the game printing a happy
    # "[REC] Recording..." message while nothing is being written for the
    # whole session. This thread (1) confirms an output file actually appears
    # and grows, and (2) falls back to OpenCV video-only if ffmpeg dies
    # mid-run, so a late failure costs seconds, not the whole day.
    def _start_watchdog(self):
        t = threading.Thread(target=self._ffmpeg_watchdog,
                             args=(self._proc,), daemon=True)
        t.start()

    def _ffmpeg_watchdog(self, proc):
        started   = time.time()
        confirmed = False
        while not self._stopped:
            time.sleep(2.0)
            if self._stopped or self._proc is not proc:
                return                      # stopped or replaced; not ours
            # 1. Did ffmpeg die after the startup check passed?
            if proc.poll() is not None:
                if self._stopped:
                    return
                print("=" * 70)
                print(f"[REC] ERROR: ffmpeg DIED after "
                      f"{time.time() - started:.0f}s (exit code "
                      f"{proc.returncode}). See "
                      f"{os.path.join(self.out_dir, 'ffmpeg.log')} for why.")
                print("[REC] Falling back to OpenCV video-only so the rest "
                      "of the session is still captured.")
                print("=" * 70)
                self._proc = None
                self.backend = None
                try:
                    self._start_opencv()
                except Exception as e:
                    print(f"[REC] OpenCV fallback also failed: {e} — "
                          f"NO RECORDING for this session!")
                return
            # 2. Positive confirmation: a session_* file exists and is growing.
            if not confirmed:
                try:
                    fresh = [
                        os.path.join(self.out_dir, f)
                        for f in os.listdir(self.out_dir)
                        if f.startswith("session_") and f.endswith(".mp4")
                        and os.path.getmtime(os.path.join(self.out_dir, f))
                            >= started - 2
                    ]
                except OSError:
                    fresh = []
                sizes = [os.path.getsize(p) for p in fresh
                         if os.path.exists(p)]
                if any(s > 0 for s in sizes):
                    confirmed = True
                    biggest = max(zip(sizes, fresh))
                    print(f"[REC] CONFIRMED writing -> {biggest[1]} "
                          f"({biggest[0] / 1024:.0f} KB and growing)")
                elif time.time() - started > 15:
                    print("=" * 70)
                    print(f"[REC] WARNING: ffmpeg is running but NO output "
                          f"file has appeared in {os.path.abspath(self.out_dir)} "
                          f"after 15s. Check "
                          f"{os.path.join(self.out_dir, 'ffmpeg.log')}.")
                    print("=" * 70)
                    confirmed = True        # warn once, keep watching for death

    # Copy a standard Windows font into the output folder as stamp_font.ttf
    # and reference it RELATIVELY (ffmpeg runs with cwd=out_dir). This
    # sidesteps drawtext's filter-parser escaping entirely: an absolute
    # Windows path contains a drive-letter colon, and ':' is drawtext's
    # option separator — the escaping rules for it differ across ffmpeg
    # versions/parse levels and are easy to get fatally wrong. A bare
    # relative filename has no special characters at all.
    def _ensure_stamp_font(self):
        dst = os.path.join(self.out_dir, "stamp_font.ttf")
        if os.path.isfile(dst):
            return "stamp_font.ttf"
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for f in ("consola.ttf", "arial.ttf", "calibri.ttf", "segoeui.ttf"):
            src = os.path.join(windir, "Fonts", f)
            if os.path.isfile(src):
                try:
                    shutil.copyfile(src, dst)
                    return "stamp_font.ttf"
                except Exception:
                    continue
        return None

    def _build_ffmpeg_cmd(self, input_args, venc=None, stamp=True):
        """Assemble the ffmpeg command (input_args let tests swap in lavfi)."""
        # The timestamp overlay uses ffmpeg's drawtext filter. Without an
        # explicit fontfile, drawtext asks FONTCONFIG to pick a font — and
        # Windows ffmpeg builds (e.g. the winget/Gyan build) ship with no
        # fontconfig configuration, so the filter fails to initialize and
        # ffmpeg EXITS shortly after start ("Fontconfig error: Cannot load
        # default config file"). That killed every recording silently.
        # On Windows we therefore pass an explicit, relative fontfile (see
        # _ensure_stamp_font). If stamp=False, skip the overlay entirely —
        # audio+video with no burned-in clock beats video-only.
        vf = None
        if self.timestamp and stamp:
            font_opt = ""
            if _IS_WIN:
                fontfile = self._ensure_stamp_font()
                if fontfile:
                    font_opt = f"fontfile={fontfile}:"
                else:
                    print("[REC] WARNING: no standard Windows font found — "
                          "recording WITHOUT the burned-in timestamp "
                          "(drawtext would crash ffmpeg via fontconfig).")
                    font_opt = None
            if font_opt is not None:
                vf = (f"drawtext={font_opt}text='%{{localtime}}':x=10:y=h-30:"
                      f"fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5")
        cmd = [_FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "warning",
               *input_args]
        if vf:
            cmd += ["-vf", vf]
        # Video encoder: hardware (Intel QuickSync) offloads H.264 off the CPU
        # so the detection camera loop stays responsive. Falls back to a light
        # software preset. Override with RECORD_VENC=libx264 (etc.) if needed.
        venc = venc or _DEF_VENC
        if venc == "h264_qsv":
            cmd += ["-c:v", "h264_qsv", "-preset", "veryfast",
                    "-global_quality", "25", "-pix_fmt", "nv12",
                    "-g", str(int(self.fps * 2))]
        else:
            cmd += ["-c:v", venc, "-preset", _DEF_VPRESET, "-pix_fmt", "yuv420p",
                    "-g", str(int(self.fps * 2))]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
        # Force CONSTANT-frame-rate output. dshow/avfoundation webcams deliver
        # a VARIABLE real rate (often 15-25 fps under encoder load), but the
        # input is tagged as a fixed 30 fps. With no rate control, ffmpeg
        # stamps whatever frames arrive as 30 fps, so playback runs fast when
        # the camera is slower than 30 (and slow during bursts) — that's the
        # "speed keeps changing" bug. -fps_mode cfr + -r rebuilds a true
        # constant rate from the real capture timestamps (duplicating/dropping
        # frames as needed), and aresample=async keeps audio locked to it.
        cmd += ["-fps_mode", "cfr", "-r", str(int(self.fps)),
                "-af", "aresample=async=1000"]
        if self.segment_min > 0:
            out = os.path.join(self.out_dir, "session_%Y%m%d_%H%M%S.mp4")
            cmd += [
                "-f", "segment",
                "-segment_time", str(self.segment_min * 60),
                "-reset_timestamps", "1",
                "-strftime", "1",
                # Make each individual segment a fragmented mp4 too, not just
                # the single-file mode below. Without this, a chunk only
                # becomes a valid/playable file once ffmpeg cleanly closes
                # it — so any session shorter than segment_min that ends via
                # a hard kill (closing the console window, Task Manager "End
                # Task", a crash) produces NO usable video at all, since the
                # in-progress chunk never finalizes. This flag keeps each
                # chunk continuously playable as it's written.
                "-segment_format_options", "movflags=+frag_keyframe+empty_moov",
                out,
            ]
        else:
            # Single file: fragmented mp4 so a hard kill still leaves it playable.
            out = os.path.join(
                self.out_dir,
                f"session_{datetime.now():%Y%m%d_%H%M%S}.mp4")
            cmd += ["-movflags", "+frag_keyframe+empty_moov", out]
            self._paths.append(out)
        return cmd

    def _stop_ffmpeg(self):
        if self._proc is None:
            return
        try:
            # Graceful stop so ffmpeg finalizes the file (writes the moov/trailer).
            # Windows doesn't support SIGINT via send_signal, so the portable way
            # is to send "q" to ffmpeg's stdin — its built-in "quit & finalize"
            # command. Fall back to SIGINT on Unix, then terminate/kill.
            stopped = False
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(b"q")
                    self._proc.stdin.flush()
                    stopped = True
            except Exception:
                stopped = False
            if not stopped and os.name != "nt":
                try:
                    self._proc.send_signal(signal.SIGINT)
                    stopped = True
                except Exception:
                    stopped = False
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            self._proc = None
            try:
                if self._logf:
                    self._logf.close()
            except Exception:
                pass

    # -- OpenCV backend (video only, fallback) --------------------------------
    def _start_opencv(self):
        # Use DirectShow on Windows so this index means the same physical
        # camera that camera_select.py resolved it from (pygrabber/DirectShow
        # enumeration order can differ from OpenCV's default MSMF backend —
        # the same fix already applied in preview_recording_camera.py).
        if _IS_WIN:
            self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.cam_index)
        if not self._cap.isOpened():
            self._cap = None
            msg = (f"[REC] Could NOT open recording camera at OpenCV index "
                   f"{self.cam_index}. No video recorded! Run "
                   f"`python3 session_recorder.py --list` to find the index.")
            if self.fail_hard:
                raise RuntimeError(msg)
            print("=" * 70 + f"\n{msg}\n" + "=" * 70)
            return
        if self.size and "x" in self.size:
            w, h = (int(x) for x in self.size.lower().split("x"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(5):
            self._cap.read()

        self.backend = "opencv"
        self._stop.clear()
        self._thread = threading.Thread(target=self._opencv_run, daemon=True)
        self._thread.start()
        print(f"[REC] Recording (OpenCV, video only) -- camera index "
              f"{self.cam_index}, {self.fps:g} fps -> {self.out_dir}"
              + (f" (new file every {self.segment_min} min)"
                 if self.segment_min else " (single file)"))

    def _opencv_run(self):
        interval   = 1.0 / self.fps
        next_write = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if self._writer is None or self._segment_due():
                self._open_writer(frame)
            now = time.perf_counter()
            if now >= next_write:
                self._write_frame(frame)
                next_write += interval
                if next_write < now:
                    next_write = now + interval

    def _segment_due(self):
        return (self.segment_min > 0
                and (time.time() - self._seg_start) >= self.segment_min * 60)

    def _open_writer(self, frame):
        self._release_writer()
        h, w = frame.shape[:2]
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"session_{ts}.mp4")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            path = os.path.join(self.out_dir, f"session_{ts}.avi")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, (w, h))
        self._writer    = writer
        self._seg_start = time.time()
        self._paths.append(path)
        print(f"[REC] Writing -> {path}  ({w}x{h})")

    def _write_frame(self, frame):
        if self.timestamp:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, stamp, (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, stamp, (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if self._writer is not None:
            self._writer.write(frame)

    def _release_writer(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

    def _stop_opencv(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._release_writer()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- shared ---------------------------------------------------------------
    def _report(self):
        # ffmpeg strftime files aren't tracked in _paths; list the folder.
        files = sorted(
            f for f in os.listdir(self.out_dir)
            if f.startswith("session_") and f.endswith((".mp4", ".avi"))
        ) if os.path.isdir(self.out_dir) else []
        if files:
            last = os.path.join(self.out_dir, files[-1])
            size_mb = os.path.getsize(last) / 1e6 if os.path.exists(last) else 0
            print(f"[REC] Recording saved -> {last} ({size_mb:.1f} MB)"
                  + (f"  (+{len(files)-1} earlier chunk(s) in {self.out_dir})"
                     if len(files) > 1 else ""))
        else:
            print("[REC] Nothing was recorded.")


# -- standalone helpers: list devices / test a clip ---------------------------
def _list_ffmpeg_avfoundation():
    if _HAS_FF and _IS_WIN:
        print("ffmpeg DirectShow devices (compare against RECORD_WIN_VIDEO / "
              "RECORD_WIN_AUDIO — names must match EXACTLY, including any "
              "truncation Windows applies):")
        # Prints the device list to stderr and exits non-zero -- normal.
        subprocess.run(
            [_FFMPEG, "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            stdin=subprocess.DEVNULL)
        print(f"\nConfigured video: '{_DEF_WIN_VIDEO}'"
              f"\nConfigured audio: '{_DEF_WIN_AUDIO}'")
        return
    if not (_HAS_FF and _IS_MAC):
        print("(ffmpeg device listing needs ffmpeg installed; "
              f"ffmpeg found: {_HAS_FF})")
        return
    print("ffmpeg avfoundation devices (look for VIDEO + AUDIO indices):")
    # This command prints the device list to stderr and exits non-zero -- normal.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        stdin=subprocess.DEVNULL)


def _list_opencv_cameras(max_index=6):
    print("\nOpenCV cameras (for the video-only fallback):")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index {i}: OPEN ({w}x{h}) first-read={'ok' if ok else 'FAILED'}")
            found.append(i)
        cap.release()
    print(f"  working: {found or 'none'}   (detection cam is 0 -- record on another)")
    return found


def _test_clip(seconds=5):
    print(f"\nRecording a {seconds}s test clip ...")
    rec = SessionRecorder(out_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "recordings_test"),
        segment_minutes=0)
    rec.start()
    if rec.backend is None:
        return
    time.sleep(seconds)
    rec.stop()
    print("Open the file above and confirm it shows the RIGHT camera and has sound.")


if __name__ == "__main__":
    args = sys.argv[1:]
    _list_ffmpeg_avfoundation()
    _list_opencv_cameras()
    if "--list" not in args:
        _test_clip()