"""
preview_recording_camera.py — live preview of the RECORDING camera (the j5),
so you can set its angle/framing before starting a game.

Usage:
    python preview_recording_camera.py

    A window opens showing the recording camera's live feed. Aim the camera,
    then press  q  or  Esc  to close.  Run this BEFORE launching game.py —
    the game and this preview can't share the camera at the same time.

Notes:
    * Uses the same name-based selection as the game (camera_select.py), so it
      always previews the RECORDING camera, not the detection one.
    * If camera_select isn't importable for some reason, falls back to the
      RECORD_CAM_INDEX env var, else index 0.
"""

import os
import cv2


def _recording_index():
    # Prefer the shared selector so this matches what the game records with.
    try:
        from camera_select import recording_index
        return recording_index()
    except Exception as e:
        print(f"[preview] camera_select unavailable ({e}); "
              f"falling back to RECORD_CAM_INDEX / 0")
        try:
            return int(os.environ.get("RECORD_CAM_INDEX", "0"))
        except ValueError:
            return 0


def main():
    idx = _recording_index()
    print(f"[preview] opening recording camera at index {idx} ...")

    # DirectShow opens fast and reliably on Windows.
    if os.name == "nt":
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        print(f"[preview] ERROR: could not open camera index {idx}. "
              f"Is another program (the game?) already using it?")
        return

    # Match the recorder's default capture size so framing matches the real
    # recording. (Recorder uses the camera's default; 640x480 is typical.)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win = "Recording camera preview  —  press Q or Esc to close"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[preview] live. Aim the camera, then press Q or Esc in the window.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[preview] dropped a frame; retrying ...")
            continue

        # Helpful on-screen guides for framing.
        h, w = frame.shape[:2]
        # center cross-hair
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
        cv2.putText(frame, "RECORDING CAMERA (players)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "RECORDING CAMERA (players)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or Esc
            break
        # If the user clicks the window's X, waitKey keeps returning 255 but the
        # window is gone; detect that and exit cleanly.
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[preview] closed.")


if __name__ == "__main__":
    main()
