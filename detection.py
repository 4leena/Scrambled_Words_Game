import cv2
import numpy as np
import os
import time

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "custom_markers")

MATCH_THRESHOLD = 0.25
GRID_SIZE = 9
IGNORE_BORDER = 1
ADD_MIRRORS = True

# Intermediate probe tilts. Template rotations cover 90-degree steps, these
# cover the angles in between -> ALL card angles work.
PROBE_ROTATIONS = (0, -15, 15, -30, 30, 45)

# Rotation can make one letter look like another (N rotated = Z, I = H...).
# When the game tells us which letters are in this round, prefer those:
# if the best overall match is NOT a round letter but a round letter scores
# within this margin, the round letter wins. A real intruder card matches
# clearly better than the margin and is still reported.
PREFER_DELTA = 0.06

# -----------------------------
# RESOLUTION-ADAPTIVE DETECTION CONFIG
# -----------------------------
# ArUco detection at high resolution (e.g. 1080p) flickers because fixed
# pixel-sized thresholding windows (adaptiveThreshWinSize*, block sizes) no
# longer straddle marker edges correctly once markers are rendered at 3-5x
# the pixel size they'd be at low-res. Two strategies fix this:
#   1. Downscale the frame before detection so marker pixel size stays in
#      the range the fixed windows were tuned for, then scale corners back up.
#   2. Make the custom letter-tile adaptive threshold scale its block size
#      with frame resolution instead of using a fixed value.
#
# Reference resolution: tuning below (window sizes, area floors) was
# validated against frames in roughly the 640-720px range. Frames larger
# than DETECT_DOWNSCALE_TRIGGER are downscaled by DETECT_SCALE_FACTOR
# before detection.
DETECT_DOWNSCALE_TRIGGER = 1000   # max(h, w) above this triggers downscaling
DETECT_SCALE_FACTOR = 0.5         # 1080p -> ~960x540 for detection purposes


def _get_detect_scale(frame):
    """Return the scale factor to use for this frame. 1.0 = no downscaling."""
    h, w = frame.shape[:2]
    if max(h, w) > DETECT_DOWNSCALE_TRIGGER:
        return DETECT_SCALE_FACTOR
    return 1.0


# -----------------------------
# MARKER ID DICTIONARY
# -----------------------------
def create_marker_id_dict():
    marker_dict = {}
    files = sorted([f for f in os.listdir(TEMPLATE_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for idx, file in enumerate(files):
        label = os.path.splitext(file)[0]
        marker_dict[label] = idx
    return marker_dict


# -----------------------------
# TEMPLATE LOADING
# -----------------------------
def load_templates():
    templates = {}
    for file in os.listdir(TEMPLATE_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        label = os.path.splitext(file)[0]
        path = os.path.join(TEMPLATE_DIR, file)
        img = cv2.imread(path)
        if img is None:
            continue
        norm = normalize_marker_image(img)
        bits, mask = image_to_bits(norm, GRID_SIZE, ignore_border=IGNORE_BORDER)
        rots = []
        grid_shape = (GRID_SIZE, GRID_SIZE)
        bits_grid = bits.reshape(grid_shape)
        mask_grid = mask.reshape(grid_shape)
        for k in range(4):
            rb_grid = np.rot90(bits_grid, k)
            rm_grid = np.rot90(mask_grid, k)
            angle = (k * 90) % 360
            rots.append((rb_grid.flatten(), rm_grid.flatten(), angle, 'none'))
            if ADD_MIRRORS:
                rots.append((np.fliplr(rb_grid).flatten(), np.fliplr(rm_grid).flatten(), angle, 'hflip'))
                rots.append((np.flipud(rb_grid).flatten(), np.flipud(rm_grid).flatten(), angle, 'vflip'))
        templates[label] = rots
        print(f"Loaded template: {label}")
    if not templates:
        raise ValueError(f"No templates found in {TEMPLATE_DIR}")
    return templates


def normalize_marker_image(img, size=250):
    try:
        marker = find_marker(img)
        if marker is not None:
            warped = warp_marker(img, marker)
            if warped is not None:
                return cv2.resize(warped, (size, size))
    except Exception:
        pass
    h, w = img.shape[:2]
    side = min(h, w)
    startx = (w - side) // 2
    starty = (h - side) // 2
    crop = img[starty:starty + side, startx:startx + side]
    return cv2.resize(crop, (size, size))


def image_to_bits(img, grid_size=7, ignore_border=0):
    """Convert a normalized marker image to a flattened bit grid + mask.

    The morphological opening kernel was reduced from (3,3) to (2,2). A
    (3,3) opening can fully erase strokes that are only 2-3px wide after
    resizing to 250x250 -- this was the main suspect for F being read as L,
    since F's middle horizontal bar is thin and exactly the kind of
    feature a (3,3) open kernel removes, collapsing F's bit pattern down to
    something that looks like L (which only has the bottom bar, which is
    much thicker/longer and survives).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.medianBlur(thresh, 3)
    kernel = np.ones((2, 2), np.uint8)   # was (3,3) -- gentler on thin strokes (e.g. F's middle bar)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    small = cv2.resize(thresh, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    bits = (small > 127).astype(np.uint8)
    mask = np.ones_like(bits, dtype=np.uint8)
    if ignore_border > 0:
        if ignore_border * 2 >= grid_size:
            ignore_border = max(0, grid_size // 2 - 1)
        mask[:ignore_border, :] = 0
        mask[-ignore_border:, :] = 0
        mask[:, :ignore_border] = 0
        mask[:, -ignore_border:] = 0
    return bits.flatten(), mask.flatten()


def prepare_probe_image(img, size=250):
    h, w = img.shape[:2]
    if h != w:
        side = min(h, w)
        startx = (w - side) // 2
        starty = (h - side) // 2
        img = img[starty:starty + side, startx:startx + side]
    return cv2.resize(img, (size, size))


def rotate_image(img, angle):
    if angle == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def is_square_candidate(pts, max_skew=0.35, min_area_ratio=0.5):
    rect = order_points(pts)
    d01 = np.linalg.norm(rect[1] - rect[0])
    d12 = np.linalg.norm(rect[2] - rect[1])
    d23 = np.linalg.norm(rect[3] - rect[2])
    d30 = np.linalg.norm(rect[0] - rect[3])
    side_lengths = np.array([d01, d12, d23, d30], dtype=np.float32)
    if np.min(side_lengths) < 5:
        return False
    if np.min(side_lengths) / np.max(side_lengths) < min_area_ratio:
        return False
    edges = [rect[1] - rect[0], rect[2] - rect[1], rect[3] - rect[2], rect[0] - rect[3]]
    for i in range(4):
        v1 = edges[i]
        v2 = edges[(i + 1) % 4]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        if np.abs(cos_angle) > max_skew:
            return False
    return True


def order_points(pts):
    pts = pts.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def is_valid_marker_patch(img, grid_size=GRID_SIZE, ignore_border=IGNORE_BORDER):
    if img is None or img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, stddev = cv2.meanStdDev(gray)
    if stddev[0][0] < 30:
        return False
    bits, mask = image_to_bits(img, grid_size=grid_size, ignore_border=ignore_border)
    if mask.sum() == 0:
        return False
    bits = bits.reshape((grid_size, grid_size))
    if ignore_border > 0:
        bits = bits[ignore_border:-ignore_border, ignore_border:-ignore_border]
    if bits.size == 0:
        return False
    transitions = np.sum(bits[:, :-1] != bits[:, 1:]) + np.sum(bits[:-1, :] != bits[1:, :])
    return transitions >= 10


def adaptive_marker_threshold(frame):
    """Adaptive threshold for custom letter-tile detection.

    Block size now scales with frame resolution instead of using a fixed
    value of 11. A fixed 11px window is fine at ~640px-wide frames but is
    far too small at 1080p+ -- it ends up landing entirely inside a single
    black or white region of a marker instead of straddling the edge,
    which is the root cause of frame-to-frame flicker at high resolution.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = frame.shape[:2]
    # Scale block size with the smaller frame dimension; must be odd and >= 3.
    block_size = max(11, int(min(h, w) * 0.018) | 1)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 7)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thresh


def are_markers_too_close(marker_a, marker_b, min_center_distance=40):
    a = get_marker_centroid(marker_a)
    b = get_marker_centroid(marker_b)
    return np.linalg.norm(a - b) < min_center_distance


def filter_close_markers(markers, min_center_distance=40):
    filtered = []
    for marker in markers:
        if not any(are_markers_too_close(marker, existing, min_center_distance)
                   for existing in filtered):
            filtered.append(marker)
    return filtered


def find_markers(frame, max_markers=None):
    """Find candidate letter-tile squares in frame.

    Area floor now scales with frame resolution. Previously this used
    `area < 3000 or area < frame_area * 0.001`, which mixes a fixed pixel
    floor with a resolution-relative floor -- at high resolution the two
    clauses disagree right around the boundary, causing tiles near the
    threshold to pass/fail inconsistently between frames (flicker). Using a
    single resolution-relative floor removes that inconsistency.
    """
    thresh = adaptive_marker_threshold(frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = max(800.0, frame_area * 0.0015)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0 or w == 0:
            continue
        # NOTE: w/h are the AXIS-ALIGNED bounding box. A 45-degree tilted
        # square fills only ~50% of it, so these limits must leave room for
        # tilt. True square-ness is still enforced by is_square_candidate.
        if float(max(w, h)) / float(min(w, h)) > 1.5:
            continue
        if area / float(w * h) < 0.5:
            continue
        if not is_square_candidate(approx, max_skew=0.30, min_area_ratio=0.65):
            continue
        warped = warp_marker(frame, approx)
        if warped is None or not is_valid_marker_patch(warped):
            continue
        candidates.append((area, approx))
    candidates.sort(key=lambda item: item[0], reverse=True)
    markers = [approx for _, approx in candidates]
    markers = filter_close_markers(markers)
    if max_markers is not None:
        markers = markers[:max_markers]
    return markers


def get_marker_centroid(marker):
    pts = marker.reshape(-1, 2)
    return np.mean(pts, axis=0)


def sort_markers_by_position(markers, row_threshold=30):
    """
    Sort markers left-to-right, top-to-bottom.
    Groups markers by similar y-coordinate (within row_threshold) into rows,
    then sorts each row left-to-right, and rows top-to-bottom.
    """
    if not markers:
        return []

    positions = []
    for marker in markers:
        center = get_marker_centroid(marker)
        positions.append((center[0], center[1], marker))  # (x, y, marker)

    positions.sort(key=lambda item: item[1])

    rows = []
    current_row = [positions[0]]

    for i in range(1, len(positions)):
        x, y, marker = positions[i]
        prev_y = current_row[-1][1]

        if abs(y - prev_y) < row_threshold:
            current_row.append(positions[i])
        else:
            rows.append(current_row)
            current_row = [positions[i]]

    rows.append(current_row)

    sorted_markers = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        sorted_markers.extend([marker for _, _, marker in row])

    return sorted_markers


def find_marker(frame):
    markers = find_markers(frame, max_markers=1)
    return markers[0] if markers else None


def warp_marker(frame, pts):
    try:
        if pts is None:
            return None
        pts = pts.reshape(-1, 2)
        if len(pts) != 4:
            return None
        rect = order_points(pts)
        size = 250
        dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (size, size))
    except Exception:
        return None


def classify(frame, templates, prefer=None):
    """Original rotation/mirror template matching, extended with intermediate
    probe tilts (all angles) and an optional 'prefer' set of round letters to
    resolve rotation aliases (N/Z, I/H, M/W) without phantom detections."""
    probe = prepare_probe_image(frame)

    # Probe bits at each tilt, computed once.
    probe_variants = []
    for rot in PROBE_ROTATIONS:
        rp = rotate_image(probe, rot)
        probe_variants.append(image_to_bits(rp, GRID_SIZE, ignore_border=IGNORE_BORDER))

    best = None        # (ratio, dist, label, angle, transform)
    best_prefer = None

    for label, variants in templates.items():
        in_prefer = prefer is not None and label.upper() in prefer
        for tmpl_bits, tmpl_mask, angle, transform in variants:
            for probe_bits, probe_mask in probe_variants:
                if tmpl_bits.size != probe_bits.size:
                    continue
                combined_mask = (tmpl_mask.astype(bool)) & (probe_mask.astype(bool))
                ncomp = int(combined_mask.sum())
                if ncomp == 0:
                    continue
                dist = int(np.count_nonzero(tmpl_bits[combined_mask] != probe_bits[combined_mask]))
                ratio = dist / ncomp
                cand = (ratio, dist, label, angle, transform)
                if best is None or ratio < best[0]:
                    best = cand
                if in_prefer and (best_prefer is None or ratio < best_prefer[0]):
                    best_prefer = cand
        # Early exit on a near-perfect round-letter match (keeps FPS up).
        if best is not None and best[0] < 0.05 and (prefer is None or best[2].upper() in prefer):
            break

    if best is None:
        return None, 255, 1.0, None, None

    chosen = best
    if (prefer is not None and best[2].upper() not in prefer
            and best_prefer is not None
            and best_prefer[0] - best[0] <= PREFER_DELTA):
        chosen = best_prefer

    ratio, dist, label, angle, transform = chosen
    return label, dist, ratio, angle, transform


def _aruco_bboxes(frame, padding=25):
    """Return bounding boxes (x1,y1,x2,y2) of every ArUco marker in the frame,
    expanded by `padding` pixels so adjacent letter tiles are not clipped.

    Detects ALL THREE dictionaries fresh (no temporal hold) so the exclusion
    zones are exactly where the markers are THIS frame — this is what
    prevents a standard ArUco word/player card from being misread as a
    letter tile.
    """
    # Fresh detection, no temporal hold, so boxes match the current frame.
    ids_4, corners_4 = _detect_4x4(frame)
    ids_6, corners_6 = _detect_6x6(frame)
    ids_w, corners_w = _detect_word_cards(frame)
    all_corners = list(corners_4) + list(corners_6) + list(corners_w)
    if not all_corners:
        return []
    h, w = frame.shape[:2]
    boxes = []
    for mc in all_corners:
        pts = mc[0]
        x1 = max(0, int(pts[:, 0].min()) - padding)
        y1 = max(0, int(pts[:, 1].min()) - padding)
        x2 = min(w, int(pts[:, 0].max()) + padding)
        y2 = min(h, int(pts[:, 1].max()) + padding)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _overlaps_aruco(contour, aruco_boxes, overlap_threshold=0.25):
    """Return True when the contour's bounding box overlaps an ArUco bbox by
    more than `overlap_threshold` of the contour area."""
    if not aruco_boxes:
        return False
    pts = contour.reshape(-1, 2)
    mx1, my1 = int(pts[:, 0].min()), int(pts[:, 1].min())
    mx2, my2 = int(pts[:, 0].max()), int(pts[:, 1].max())
    m_area = max(1, (mx2 - mx1) * (my2 - my1))
    for (ax1, ay1, ax2, ay2) in aruco_boxes:
        ix = max(0, min(mx2, ax2) - max(mx1, ax1))
        iy = max(0, min(my2, ay2) - max(my1, ay1))
        if (ix * iy) / m_area >= overlap_threshold:
            return True
    return False


_LETTER_ACCEPT_RATIO = 0.35   # squares scoring above this on all letter templates → not a letter tile
_MAX_BOUNDARY_CANDIDATES = 8  # keep only the N smallest unrecognised squares as boundary candidates


def detect_markers(frame, templates, marker_id_dict, allowed_labels=None,
                   play_area_hull=None):
    """Detect custom letter-tile markers in `frame`.

    Returns (letter_detections, boundary_pts).

    Strategy — inverse classification:
      • Every detected square is classified against letter templates only.
      • If it scores ratio < _LETTER_ACCEPT_RATIO  → letter tile.
      • If it scores ratio ≥ _LETTER_ACCEPT_RATIO  → NOT a letter; collected as
        a boundary/ArUco candidate.
    This works because custom letter tiles reliably match letter templates,
    while ArUco grid patterns do not match any letter template well.
    After collection, boundary candidates are capped to the N smallest
    (by contour area) to discard large misclassified tiles that might land
    outside the play area.

    If `play_area_hull` is provided, any square whose centre falls OUTSIDE
    the hull is skipped before classification. This is the primary guard that
    stops a word/ID card held up to the camera (outside the play area) from
    ever being considered as a letter tile.
    """
    aruco_boxes  = _aruco_bboxes(frame)
    letter_tmpls = {k: v for k, v in templates.items()
                    if not k.upper().startswith("BOUNDARY")}

    prefer   = {str(l).upper() for l in allowed_labels} if allowed_labels else None
    markers  = sort_markers_by_position(find_markers(frame))
    detected = []
    boundary_candidates = []   # list of (area, cx, cy)

    for marker in markers:
        # Skip squares outside the play area entirely — a card held up to the
        # camera above the boundary is never classified as a letter.
        if play_area_hull is not None:
            c = marker.reshape(-1, 2)
            cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
            inside = cv2.pointPolygonTest(
                play_area_hull.astype(np.float32), (cx, cy), False) >= 0
            if not inside:
                continue

        if _overlaps_aruco(marker, aruco_boxes):
            continue

        warped = warp_marker(frame, marker)
        if warped is None:
            continue

        label, dist, ratio, angle, transform = classify(warped, letter_tmpls, prefer=prefer)
        corners = order_points(marker)

        if label is not None and ratio < _LETTER_ACCEPT_RATIO:
            # ── Confirmed letter tile ─────────────────────────────────
            valid_patch = is_valid_marker_patch(warped)
            marker_id   = marker_id_dict.get(label, "?") if label is not None else "?"
            detected.append({
                "corners":   corners,
                "label":     label,
                "id":        marker_id,
                "valid":     valid_patch and ratio < MATCH_THRESHOLD,
                "distance":  dist,
                "ratio":     ratio,
                "angle":     angle,
                "transform": transform,
                "warped":    warped,
            })
        else:
            # ── Does not look like a letter → boundary/ArUco candidate ─
            area = float(cv2.contourArea(corners.astype(np.float32)))
            cx   = float(corners[:, 0].mean())
            cy   = float(corners[:, 1].mean())
            boundary_candidates.append((area, cx, cy))

    # Prefer smallest candidates — boundary corner markers are physically smaller
    # than letter tiles, so this filters out large misclassified tiles.
    boundary_candidates.sort(key=lambda t: t[0])
    boundary_pts = [(cx, cy)
                    for _, cx, cy in boundary_candidates[:_MAX_BOUNDARY_CANDIDATES]]

    return detected, boundary_pts

# -------------------------------------------------------
# ARUCO DETECTOR — auto-selects the dictionary that finds your markers
# -------------------------------------------------------
# Set BOUNDARY_MARKER_IDS to the IDs printed on your 4 corner markers.
# Player ID cards must use IDs NOT in this set.
BOUNDARY_MARKER_IDS = {10, 11, 12, 13}   # corner boundary markers

# Corner markers for the dedicated WORD-CARD placement zone — a separate,
# smaller area (like a little "in-tray") where the player places the word
# card. Same physical family as the play-area corners (DICT_4X4_250) but a
# different, non-overlapping set of IDs so the two zones never get confused.
CARD_AREA_MARKER_IDS = {14, 15, 16, 17}   # corner markers for the card zone

# Everything detected by the 4x4 dictionary that we actually care about —
# play-area corners AND card-area corners. _detect_4x4 keeps anything in
# this combined set; individual callers (detect_play_area / detect_card_area)
# each filter down to their own subset afterwards.
_ALL_4X4_MARKER_IDS = BOUNDARY_MARKER_IDS | CARD_AREA_MARKER_IDS

# -------------------------------------------------------
# WORD CARD DETECTION — DICT_6X6_250
# -------------------------------------------------------
# Each level has a deck of printed cards. Word cards are printed using the
# ORIGINAL ArUco dictionary (DICT_ARUCO_ORIGINAL, 5x5 interior bits) — a
# DIFFERENT physical dictionary from the 6x6 family used by player cards.
# IDs 0-199 are reserved for word cards within DICT_ARUCO_ORIGINAL; player
# cards use IDs 200-999 within DICT_6X6_1000. Two different dictionaries,
# so there is zero risk of a player card being misread as a word card (or
# vice versa) even though both happen to use "ID-range" bookkeeping.
#
# ID layout:
#   Boundary corners :  DICT_4X4_250        IDs 10-13
#   Word cards       :  DICT_ARUCO_ORIGINAL IDs  0-199
#     Level 1 Animals:    0 –  8  (9 words)
#     Level 2 Tech:      10 – 41  (32 words)
#     Level 3 Galaxies:  50 – 73  (24 words)
#   Player ID cards  :  DICT_6X6_1000       IDs 200-999

WORD_CARD_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
_word_card_params = cv2.aruco.DetectorParameters()
_word_card_params.adaptiveThreshWinSizeMin    = 3
_word_card_params.adaptiveThreshWinSizeMax    = 53
_word_card_params.adaptiveThreshWinSizeStep   = 10
_word_card_params.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
# Match the tuning used for boundary/player detection (_ARUCO_PARAMS below)
# instead of leaving these at OpenCV's stricter defaults (0.03 / 0.03),
# which was previously making word cards need to be held closer/flatter
# than player cards to register at all.
_word_card_params.polygonalApproxAccuracyRate = 0.08
# Anti-ghost tuning. DICT_ARUCO_ORIGINAL is a loose dictionary (1024
# markers, weak separation) and the custom letter tiles are themselves
# black-bordered square grids — a tile on the table can consistently
# misdecode as a valid word-card ID. Real cards are large, flat, cleanly
# printed and decode with ~zero bit errors, so:
#   • minMarkerPerimeterRate 0.06: only card-sized markers qualify —
#     tile-sized patterns on the table are ignored outright.
#   • errorCorrectionRate 0.2 (default 0.6): misdecodes NEED the error-
#     correction slack; clean prints don't. Starve the false positives.
#   • maxErroneousBitsInBorderRate 0.2 (default 0.35): require a clean
#     quiet-zone border as well.
_word_card_params.minMarkerPerimeterRate       = 0.06
_word_card_params.errorCorrectionRate          = 0.2
_word_card_params.maxErroneousBitsInBorderRate = 0.2

# Word cards are printed in DICT_ARUCO_ORIGINAL, IDs 0-199. We detect with
# ONLY this dictionary and accept ONLY IDs in WORD_ID_RANGE, so neither
# boundary markers (4x4) nor player cards (6x6, 200+) — which live in
# entirely separate dictionaries — can ever be misread as a word card.
_WORD_CARD_DETECTORS = [
    cv2.aruco.ArucoDetector(WORD_CARD_DICT, _word_card_params),
]

# Mapping: ArUco ID → word (populated from game.py CATEGORIES at startup)
# Format: { id: "word" }
WORD_CARD_MAP: dict[int, str] = {}

# Hold-over so a single dropped frame doesn't kill the detection. This is
# deliberately small — it only bridges a momentary miss of an ALREADY
# confirmed marker, it is NOT a debounce/stability mechanism. (It used to be
# 8 frames, which meant a single misread of an ID could coast on hold-over
# for 8 frames and single-handedly satisfy game.py's own 8-consecutive-frame
# stability check, since that check operates on this function's already-
# smoothed output rather than the raw per-frame read — i.e. one false
# positive here could fake out both debounce layers stacked on top of it.)
_WORD_CARD_HOLD_FRAMES = 2

# Require the same ID to be read on >=2 raw frames before trusting it at all.
# Filters one-off misreads (e.g. a marker from a different dictionary, or
# print noise, occasionally decoding as a "valid" word-card ID) instead of
# accepting on the very first raw sighting.
_WORD_CARD_MIN_RAW_HITS = 4   # was 2 — a ghost must now survive 4 raw
                              # frames under the stricter decode settings
                              # above; a real card confirms in ~0.5s

_word_card_last_seen: dict[int, dict] = {}   # id -> {"word": str, "age": int, "hits": int}


def reset_word_card_tracking():
    """Clear all word-card detection state (raw-hit counts + hold-over).

    Call this whenever a new round starts waiting for a card, so a stale or
    falsely-confirmed marker from an earlier word can never leak into the
    next one. game.py calls this at every point `waiting_for_card` becomes
    True (start of game, after each word, new game, level change).
    """
    _word_card_last_seen.clear()


def build_word_card_map(categories: dict) -> dict[int, str]:
    """Build and register the ID→word map from the game's CATEGORIES dict.

    Call this once at game startup:
        from detection import build_word_card_map
        build_word_card_map(CATEGORIES)

    Returns the map so callers can inspect it / print it.
    """
    offsets = {"animals": 0, "tech": 10, "galaxies": 50}
    card_map: dict[int, str] = {}
    for cat, words in categories.items():
        base = offsets.get(cat, 400)
        for i, word in enumerate(words):
            card_map[base + i] = word.lower()
    WORD_CARD_MAP.clear()
    WORD_CARD_MAP.update(card_map)
    print(f"[WordCards] registered {len(card_map)} cards: "
          + ", ".join(f"{wid}={w}" for wid, w in sorted(card_map.items())))
    return card_map


def _detect_word_cards(frame):
    """Detect ONLY word-card markers (DICT_ARUCO_ORIGINAL, IDs 0-199).

    Returns (ids:list[int], corners:list[ndarray]). Mirrors _detect_4x4 /
    _detect_6x6 so all three marker families are detected and range-filtered
    the same way, and so _aruco_bboxes() can build accurate exclusion zones
    for word cards too (previously only 4x4/6x6 were excluded, which meant a
    word card held over the play area could be misread as a letter tile).
    """
    gray, scale = _to_gray(frame)
    corners, ids, _ = _WORD_CARD_DETECTORS[0].detectMarkers(gray)
    if ids is None:
        return [], []
    out_ids, out_corners = [], []
    for mid, mc in zip(ids.flatten().tolist(), corners):
        mid = int(mid)
        if mid in WORD_ID_RANGE:
            out_ids.append(mid)
            out_corners.append(mc if scale == 1.0 else mc / scale)
    return out_ids, out_corners


def detect_word_card(frame, card_area_hull=None) -> str | None:
    """Detect a word card in frame and return the word string, or None.

    Word cards are DICT_ARUCO_ORIGINAL, IDs 0-199, registered in
    WORD_CARD_MAP. This dictionary is structurally distinct from the 6x6
    family used by player cards, so a player card's bit pattern can never
    decode as a word-card ID (and vice versa) — there's no need for an
    extra range check against PLAYER_ID_RANGE the way the old (incorrect)
    shared-dictionary version needed.

    An ID must be read on >= _WORD_CARD_MIN_RAW_HITS raw frames before it's
    trusted (filters one-off misreads), and once confirmed, a momentary miss
    is bridged for only _WORD_CARD_HOLD_FRAMES frames (not a debounce layer —
    see the comment above those constants).

    Parameters
    ----------
    card_area_hull : np.ndarray or None
        Convex hull (from detect_card_area) marking the dedicated card
        placement zone. When provided, any detected word-card marker whose
        centre falls OUTSIDE the hull is ignored entirely — it never even
        reaches the raw-hit counter — so a card held up elsewhere in frame
        (or a stray reflection) can't register. When None (e.g. the zone
        markers aren't visible yet), falls back to accepting a card
        anywhere in frame, same as before this parameter existed.
    """
    seen_now: set[int] = set()
    confirmed_word: str | None = None

    id_list, id_corners = _detect_word_cards(frame)
    for mid, mc in zip(id_list, id_corners):
        if card_area_hull is not None:
            cx, cy = mc[0].mean(axis=0)
            inside = cv2.pointPolygonTest(
                card_area_hull.astype(np.float32), (float(cx), float(cy)), False) >= 0
            if not inside:
                continue
        word = WORD_CARD_MAP.get(mid)
        if not word:
            # Detected a word-card ID but it's not in the map — log it
            print(f"[WordCard] ID {mid} detected but not in WORD_CARD_MAP")
            continue
        seen_now.add(mid)
        entry = _word_card_last_seen.get(mid, {"word": word, "age": 0, "hits": 0})
        entry["word"] = word
        entry["age"] = 0
        entry["hits"] = entry.get("hits", 0) + 1
        _word_card_last_seen[mid] = entry
        if entry["hits"] >= _WORD_CARD_MIN_RAW_HITS:
            confirmed_word = word
            print(f"[WordCard] ID {mid} → {word} (confirmed, {entry['hits']} hits)")

    # Temporal hold-over — only for IDs that were already confirmed.
    stale = []
    for mid, entry in _word_card_last_seen.items():
        if mid in seen_now:
            continue
        entry["age"] += 1
        if entry["age"] > _WORD_CARD_HOLD_FRAMES:
            stale.append(mid)
        elif entry.get("hits", 0) >= _WORD_CARD_MIN_RAW_HITS and confirmed_word is None:
            confirmed_word = entry["word"]
    for mid in stale:
        del _word_card_last_seen[mid]

    return confirmed_word


def draw_word_card_overlay(frame, word: str | None) -> None:
    """Draw a small HUD on frame showing the currently detected word card."""
    if word is None:
        return
    h, w = frame.shape[:2]
    label = f"Card detected: {word.upper()}"
    cv2.putText(frame, label, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)

_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# Tuned for frames downscaled to ~DETECT_DOWNSCALE_TRIGGER * DETECT_SCALE_FACTOR
# (i.e. ~960px wide for a 1080p source). These thresholds previously caused
# flicker at full 1080p+ resolution because the fixed-pixel window sizes
# didn't straddle marker edges consistently frame-to-frame; downscaling the
# frame before detection (see _detect_all_aruco) keeps marker pixel size in
# the range these were validated against, instead of needing ever-larger
# windows that themselves introduce false edges.
_ARUCO_PARAMS.adaptiveThreshWinSizeMin    = 3
_ARUCO_PARAMS.adaptiveThreshWinSizeMax    = 53
_ARUCO_PARAMS.adaptiveThreshWinSizeStep   = 10
_ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.08
_ARUCO_PARAMS.minMarkerPerimeterRate      = 0.02
# Corner refinement smooths out small jitter frame-to-frame, which helps
# perceived flicker even when the marker is correctly detected every frame.
_ARUCO_PARAMS.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
_ARUCO_PARAMS.cornerRefinementWinSize     = 5
_ARUCO_PARAMS.cornerRefinementMaxIterations = 30
_ARUCO_PARAMS.cornerRefinementMinAccuracy = 0.05

# ════════════════════════════════════════════════════════════════════════════
# MARKER ARCHITECTURE  (single source of truth — do not reuse IDs across these)
# ════════════════════════════════════════════════════════════════════════════
#   Boundary corners :  DICT_4X4_250        IDs 10-13
#   Word cards       :  DICT_ARUCO_ORIGINAL IDs 0-199
#   Player ID cards  :  DICT_6X6_1000       IDs 200-999
#
# THREE physically different dictionaries coexist in every frame. We run a
# DEDICATED detector for each, and immediately discard any ID that falls
# outside that family's expected range. This stops a 4x4 boundary square
# from ever being misread as a word/player card (and vice versa) — the bug
# that kept making the boundary flicker or the player cards vanish — and,
# critically, means a word card's bit pattern is never even tested against
# the 6x6 dictionary (or vice versa), since DICT_ARUCO_ORIGINAL (5x5) and
# DICT_6X6_1000 (6x6) are structurally different grids, not ID slices of
# the same dictionary.
# ════════════════════════════════════════════════════════════════════════════

WORD_ID_RANGE   = range(0, 200)      # word cards: 0-199
PLAYER_ID_RANGE = range(200, 1000)   # player cards: 200-999

_aruco_detector_4x4 = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250), _ARUCO_PARAMS)
_aruco_detector_6x6 = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000), _ARUCO_PARAMS)

# Back-compat aliases (some helpers still reference these names).
_aruco_detector = _aruco_detector_4x4
_aruco_dict_locked = True

# ── Temporal smoothing ──────────────────────────────────────────────────
# Even with correct detection, single-frame dropouts (motion blur, a hand
# briefly crossing the marker, auto-exposure adjusting) read as "flicker"
# to the user even though detection is working fine 9 frames out of 10.
# Cache the last-seen position of each ID for a few frames so a momentary
# miss doesn't make the marker visually disappear.
_ARUCO_HOLD_FRAMES = 5   # how many consecutive missed frames to tolerate
_aruco_last_seen = {}    # marker_id -> {"corners": ndarray, "age": int}


def _to_gray(frame):
    """Downscale (if needed) and convert to grayscale. Returns (gray, scale)."""
    scale = _get_detect_scale(frame)
    if scale != 1.0:
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, scale


def _detect_4x4(frame):
    """Detect 4x4 markers we care about: play-area boundary corners
    (BOUNDARY_MARKER_IDS) AND word-card-area corners (CARD_AREA_MARKER_IDS).

    Returns (ids:list[int], corners:list[ndarray]). Any 4x4 marker whose ID
    isn't in either set is discarded, so stray 4x4 tags can't pollute either
    hull. Callers that need just one zone (detect_play_area /
    detect_card_area) filter this combined result down to their own subset.
    """
    gray, scale = _to_gray(frame)
    corners, ids, _ = _aruco_detector_4x4.detectMarkers(gray)
    if ids is None:
        return [], []
    out_ids, out_corners = [], []
    for mid, mc in zip(ids.flatten().tolist(), corners):
        mid = int(mid)
        if mid in _ALL_4X4_MARKER_IDS:
            out_ids.append(mid)
            out_corners.append(mc if scale == 1.0 else mc / scale)
    return out_ids, out_corners


def _detect_6x6(frame):
    """Detect ONLY 6x6 markers (word cards 0-199 and player cards 200-999).

    Returns (ids:list[int], corners:list[ndarray]). Any 6x6 marker outside
    the combined word+player ID range is discarded as a false read.
    """
    gray, scale = _to_gray(frame)
    corners, ids, _ = _aruco_detector_6x6.detectMarkers(gray)
    if ids is None:
        return [], []
    out_ids, out_corners = [], []
    for mid, mc in zip(ids.flatten().tolist(), corners):
        mid = int(mid)
        if mid in WORD_ID_RANGE or mid in PLAYER_ID_RANGE:
            out_ids.append(mid)
            out_corners.append(mc if scale == 1.0 else mc / scale)
    return out_ids, out_corners


def _detect_all_aruco(frame, use_hold=True):
    """Detect every game marker by running both dedicated detectors and
    concatenating their (already range-filtered) results.

    Each family is filtered to its own ID range inside _detect_4x4 /
    _detect_6x6, so cross-dictionary ghosts are gone before they reach the
    shared temporal-hold cache. This is what keeps the boundary stable while
    still seeing 6x6 player/word cards in the same frame.
    """
    ids_4, corners_4 = _detect_4x4(frame)
    ids_6, corners_6 = _detect_6x6(frame)
    ids_out     = ids_4 + ids_6
    corners_out = corners_4 + corners_6

    if not use_hold:
        return ids_out, corners_out

    return _apply_temporal_hold(ids_out, corners_out)


def _apply_temporal_hold(ids_out, corners_out):
    """Patch single-frame dropouts using the last-seen position of each ID.

    A marker missing for <= _ARUCO_HOLD_FRAMES consecutive frames is
    reported at its last known position instead of disappearing. A marker
    that's actually gone (removed, occluded for a while) ages out normally.
    """
    seen_now = set()
    for mid, corners in zip(ids_out, corners_out):
        mid = int(mid)
        seen_now.add(mid)
        _aruco_last_seen[mid] = {"corners": corners, "age": 0}

    final_ids, final_corners = list(ids_out), list(corners_out)
    stale = []
    for mid, entry in _aruco_last_seen.items():
        if mid in seen_now:
            continue
        entry["age"] += 1
        if entry["age"] <= _ARUCO_HOLD_FRAMES:
            final_ids.append(mid)
            final_corners.append(entry["corners"])
        else:
            stale.append(mid)
    for mid in stale:
        del _aruco_last_seen[mid]

    return final_ids, final_corners


def detect_play_area(frame, require_all=True):
    """Find the boundary ArUco markers and return the convex hull of the play area.

    Parameters
    ----------
    require_all : bool
        If True (default), only return a hull when ALL 4 boundary markers
        (BOUNDARY_MARKER_IDS) are visible. This prevents the hull from
        shrinking/skewing — and the boundary overlay from flickering —
        whenever a single corner is briefly occluded by a hand, phone, etc.
        If False, falls back to the old behaviour (hull from whatever 3+
        markers are visible), which is less stable but more permissive.

    Returns
    -------
    pts  : list of (x, y) centre-points for every detected boundary marker
    hull : np.ndarray shape (N, 2) — convex hull polygon, or None if the
           visibility requirement isn't met.
    """
    # Use the dedicated 4x4 detector — boundary markers are 4x4 only, and
    # _detect_4x4 already filters to BOUNDARY_MARKER_IDS, so nothing from the
    # 6x6 word/player dictionary can ever perturb the hull.
    id_list, corners = _detect_4x4(frame)
    if not id_list:
        return [], None

    pts = []
    for mid, mc in zip(id_list, corners):
        if int(mid) in BOUNDARY_MARKER_IDS:
            cx, cy = mc[0].mean(axis=0)
            pts.append((float(cx), float(cy)))

    min_required = len(BOUNDARY_MARKER_IDS) if require_all else 3
    if len(pts) < min_required:
        return pts, None

    hull = cv2.convexHull(np.array(pts, dtype=np.float32))
    return pts, hull.reshape(-1, 2)


def detect_card_area(frame, require_all=True):
    """Find the 4 corner markers of the word-card placement zone and return
    the convex hull of that (smaller, separate) area.

    Mirrors detect_play_area exactly, but filters to CARD_AREA_MARKER_IDS
    instead of BOUNDARY_MARKER_IDS, so the two zones are computed
    independently even though both live in the same 4x4 dictionary.

    Parameters
    ----------
    require_all : bool
        If True (default), only return a hull when ALL 4 card-area corner
        markers are visible, so the zone doesn't shrink/skew when a hand
        briefly covers one corner. If False, a hull from 3+ markers is
        returned instead (less stable, more permissive).

    Returns
    -------
    pts  : list of (x, y) centre-points for every detected card-area marker
    hull : np.ndarray shape (N, 2) — convex hull polygon, or None if the
           visibility requirement isn't met.
    """
    id_list, corners = _detect_4x4(frame)
    if not id_list:
        return [], None

    pts = []
    for mid, mc in zip(id_list, corners):
        if int(mid) in CARD_AREA_MARKER_IDS:
            cx, cy = mc[0].mean(axis=0)
            pts.append((float(cx), float(cy)))

    min_required = len(CARD_AREA_MARKER_IDS) if require_all else 3
    if len(pts) < min_required:
        return pts, None

    hull = cv2.convexHull(np.array(pts, dtype=np.float32))
    return pts, hull.reshape(-1, 2)


def draw_card_area(frame, hull, pts):
    """Draw the card-area boundary polygon and corner dots on frame.

    Uses a distinct color (magenta) from the play-area overlay (amber) so
    the two zones are visually distinguishable to the player/operator.
    """
    if hull is not None and len(hull) >= 3:
        cv2.polylines(frame, [hull.astype(np.int32)],
                      isClosed=True, color=(200, 0, 200), thickness=2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull.astype(np.int32)], (60, 0, 60))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 8, (200, 0, 200), -1)

    n = len(pts)
    label = f"Card area: {n}/4" if n < 4 else "Card area OK"
    color = (200, 0, 200) if n == 4 else (0, 80, 255)
    cv2.putText(frame, label, (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def hull_from_points(pts):
    """Build a convex hull np.ndarray (N,2) from a list of (x,y) tuples, or None."""
    if len(pts) < 3:
        return None
    arr = np.array(pts, dtype=np.float32)
    h = cv2.convexHull(arr)
    return h.reshape(-1, 2)


def filter_markers_by_play_area(markers, hull):
    """Keep only markers whose centroid falls inside the play-area hull."""
    if hull is None:
        return markers
    kept = []
    for m in markers:
        cx, cy = m["corners"].mean(axis=0)
        inside = cv2.pointPolygonTest(
            hull.astype(np.float32), (float(cx), float(cy)), False) >= 0
        if inside:
            kept.append(m)
    return kept


def draw_play_area(frame, hull, pts, all_aruco_ids=None):
    """Draw the boundary polygon, corner dots, and ArUco ID labels on frame."""
    if hull is not None and len(hull) >= 3:
        cv2.polylines(frame, [hull.astype(np.int32)],
                      isClosed=True, color=(0, 200, 255), thickness=2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull.astype(np.int32)], (0, 40, 50))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 8, (0, 200, 255), -1)

    n = len(pts)
    label = f"Boundary: {n}/4" if n < 4 else "Boundary OK"
    color = (0, 200, 255) if n == 4 else (0, 80, 255)
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Show all detected ArUco IDs so the user can identify their corner markers
    if all_aruco_ids:
        id_str = f"ArUco IDs seen: {sorted(set(all_aruco_ids))}"
        cv2.putText(frame, id_str, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 1, cv2.LINE_AA)
    return frame


def get_all_aruco_ids(frame):
    """Return all ArUco IDs visible in the frame (for debugging)."""
    id_list, _ = _detect_all_aruco(frame)
    return [int(i) for i in id_list]


# -------------------------------------------------------
# PLAYER ID DETECTION
# -------------------------------------------------------

_aruco_print_count = 0


def detect_player_ids(frame):
    """Detect ArUco player ID cards.
    Player cards use IDs 200-999, printed in DICT_6X6_1000.
    Word cards (0-199, same dictionary) and boundary markers (10-13, 4x4)
    are excluded by the PLAYER_ID_RANGE check below.
    """
    # Use the 6x6 detector directly — player cards live there. (Going through
    # _detect_all_aruco would also work, but this keeps player detection
    # independent of boundary detection.)
    id_list, corners = _detect_6x6(frame)
    if not id_list:
        return []

    global _aruco_print_count
    if _aruco_print_count < 5 and id_list:
        print(f"[ArUco] 6x6 IDs detected: {id_list}")
        _aruco_print_count += 1

    # Only accept player-range IDs (200-999)
    player_pairs = [(pid, mc) for pid, mc in zip(id_list, corners)
                    if int(pid) in PLAYER_ID_RANGE]
    if not player_pairs:
        return []
    player_pairs.sort(key=lambda pair: pair[1][0][:, 0].mean())
    return [int(pid) for pid, _ in player_pairs]


def draw_player_markers(frame, confirmed_ids):
    if not confirmed_ids:
        return frame

    id_list, corners = _detect_6x6(frame)
    if not id_list:
        return frame

    for mid, mc in zip(id_list, corners):
        if int(mid) not in confirmed_ids:
            continue
        pts = mc[0].astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 80), thickness=3)
        top_left = pts[0]
        cv2.putText(frame, f"Player_{mid}",
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2, cv2.LINE_AA)
    return frame


# -------------------------------------------------------
# DIAGNOSTICS — run standalone to debug letter confusion (e.g. F read as L)
# -------------------------------------------------------

def debug_compare_letters(label_a, label_b, templates=None):
    """Print the 9x9 bit grids for two letter labels side-by-side, plus their
    hamming distance, so you can see exactly why two letters are colliding.

    Usage:
        import detection as det
        templates = det.load_templates()
        det.debug_compare_letters('F', 'L', templates)

    Each printed grid is one orientation (angle=0, transform='none') per
    label. A 9x9 grid is small -- if F and L look near-identical here, the
    grid resolution / morphology is erasing the feature that distinguishes
    them (almost certainly F's middle bar), and the fix has to happen on
    the card design (bolder/thicker middle bar) or by upping GRID_SIZE.
    """
    if templates is None:
        templates = load_templates()

    def first_variant(label):
        for key, variants in templates.items():
            if key.upper() == label.upper():
                for bits, mask, angle, transform in variants:
                    if angle == 0 and transform == 'none':
                        return bits.reshape(GRID_SIZE, GRID_SIZE)
        return None

    grid_a = first_variant(label_a)
    grid_b = first_variant(label_b)
    if grid_a is None or grid_b is None:
        print(f"Could not find templates for '{label_a}' and/or '{label_b}'. "
              f"Available labels: {sorted(templates.keys())}")
        return

    print(f"\n{label_a} bit grid:")
    for row in grid_a:
        print(''.join('#' if v else '.' for v in row))

    print(f"\n{label_b} bit grid:")
    for row in grid_b:
        print(''.join('#' if v else '.' for v in row))

    dist = int(np.sum(grid_a != grid_b))
    total = grid_a.size
    print(f"\nHamming distance: {dist}/{total} ({dist/total:.2%} different)")
    if dist / total < 0.15:
        print("⚠ These grids are nearly identical -- GRID_SIZE or morphology "
              "is erasing the distinguishing feature. Consider a bolder card "
              "design or increasing GRID_SIZE.")
    else:
        print("Grids are meaningfully different -- if you're still seeing "
              "F misread as L, the issue is more likely in image capture "
              "(lighting/angle/blur on the physical card) than the template "
              "itself. Try printing debug_compare_letters output using a "
              "frame captured live instead of the static template.")