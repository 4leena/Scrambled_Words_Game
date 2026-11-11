"""
camera_select.py  —  pick the right camera by DEVICE NAME.

Setup in this lab
-----------------
    DETECTION camera (aimed at the board) : "HD Pro Webcam C920"
    RECORDING camera (films the players)  : "j5 WebCam JVCU100"

Because the two cameras are now different models, they report different names,
so we can map each role to a stable name instead of a fragile OpenCV index that
Windows may reorder between runs.

Public functions
----------------
    detection_index()   -> int   OpenCV index of the detection camera
    recording_index()   -> int   OpenCV index of the recording camera
    index_for_name(sub) -> int   index of the first camera whose name contains `sub`
    list_cameras()      -> list[str]   device names in OpenCV index order

Environment overrides (win over name lookup if set)
    DETECT_CAM_INDEX    force the detection camera index
    RECORD_CAM_INDEX    force the recording camera index

If a name isn't found, we fall back to a sensible default and print a warning,
so the game still starts instead of crashing.
"""

import os

# --- Configure the two camera names here (substring match, case-insensitive) ---
DETECTION_CAMERA_NAME = "C920"                 # HD Pro Webcam C920  (board)
RECORDING_CAMERA_NAME = "j5 WebCam JVCU100"    # films the players

# Fallback indices used only if a name can't be found at all.
_DETECTION_FALLBACK = 0
_RECORDING_FALLBACK = 1


_camera_cache = None   # device-name list, enumerated once and reused


def list_cameras(force=False):
    """Return camera device names in OpenCV index order (index 0 first).

    Uses pygrabber on Windows (DirectShow). On other OSes, or if pygrabber
    isn't installed, returns an empty list and callers fall back to indices.

    The result is CACHED: pygrabber's DirectShow enumeration can take a few
    seconds, and both detection_index() and recording_index() need it, so we
    enumerate once per run. Pass force=True to re-scan (e.g. after hot-plugging
    a camera).
    """
    global _camera_cache
    if _camera_cache is not None and not force:
        return _camera_cache
    try:
        from pygrabber.dshow_graph import FilterGraph
        _camera_cache = list(FilterGraph().get_input_devices())
    except Exception:
        _camera_cache = []
    return _camera_cache


def index_for_name(substring, names=None):
    """Index of the first camera whose name contains `substring` (case-insensitive).

    Returns None if not found.
    """
    if names is None:
        names = list_cameras()
    sub = substring.lower()
    for i, name in enumerate(names):
        if sub in name.lower():
            return i
    return None


def _resolve(role_name, camera_name, fallback, env_var):
    # 1. Explicit env override wins.
    forced = os.environ.get(env_var)
    if forced not in (None, ""):
        try:
            idx = int(forced)
            print(f"[CAMERA] {role_name}: using {env_var}={idx}")
            return idx
        except ValueError:
            print(f"[CAMERA] {role_name}: ignoring bad {env_var}={forced!r}")

    # 2. Look the camera up by name.
    names = list_cameras()
    idx = index_for_name(camera_name, names)
    if idx is not None:
        print(f"[CAMERA] {role_name}: '{names[idx]}' at index {idx}")
        return idx

    # 3. Fall back so the game still runs.
    if names:
        print(f"[CAMERA] {role_name}: WARNING '{camera_name}' not found in "
              f"{names}; falling back to index {fallback}")
    else:
        print(f"[CAMERA] {role_name}: WARNING could not list cameras "
              f"(pygrabber missing?); falling back to index {fallback}")
    return fallback


def detection_index():
    """OpenCV index of the DETECTION camera (the one aimed at the board)."""
    return _resolve("detection", DETECTION_CAMERA_NAME,
                    _DETECTION_FALLBACK, "DETECT_CAM_INDEX")


def recording_index():
    """OpenCV index of the RECORDING camera (the one filming players)."""
    return _resolve("recording", RECORDING_CAMERA_NAME,
                    _RECORDING_FALLBACK, "RECORD_CAM_INDEX")


if __name__ == "__main__":
    cams = list_cameras()
    if cams:
        print("Cameras detected:")
        for i, n in enumerate(cams):
            print(f"  {i}: {n}")
    else:
        print("No cameras listed (is pygrabber installed? `pip install pygrabber`)")
    print()
    print(f"--> DETECTION index: {detection_index()}")
    print(f"--> RECORDING index: {recording_index()}")