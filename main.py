import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os

MODEL_PATH = "models/gesture_recognizer.task"

# ── MediaPipe setup ────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
GestureRecognizer     = mp.tasks.vision.GestureRecognizer
GestureRecognizerOpts = mp.tasks.vision.GestureRecognizerOptions
RunningMode           = mp.tasks.vision.RunningMode

# ── Gesture display map ────────────────────────────────────
# All 8 supported gestures with emoji + color (BGR)
GESTURE_INFO = {
    "None":        ("·  None",         (120, 120, 120)),
    "Unknown":     ("?  Unknown",      (100, 100, 100)),
    "Closed_Fist": ("✊  Closed Fist",  (0,   140, 255)),
    "Open_Palm":   ("🖐  Open Palm",    (0,   220, 120)),
    "Pointing_Up": ("☝  Pointing Up",  (255, 220,   0)),
    "Thumb_Down":  ("👎  Thumb Down",   (0,    60, 220)),
    "Thumb_Up":    ("👍  Thumb Up",     (0,   255, 100)),
    "Victory":     ("✌  Victory",      (200,   0, 255)),
    "ILoveYou":    ("🤟  I Love You",   (255,  80,   0)),
}

# ── Hand skeleton connections ──────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
FINGER_TIPS = {4, 8, 12, 16, 20}

# ── Async result store ─────────────────────────────────────
latest = {"result": None}

def on_result(result, output_image, timestamp_ms):
    latest["result"] = result

# ── Build recognizer ───────────────────────────────────────
options = GestureRecognizerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result
)
recognizer = GestureRecognizer.create_from_options(options)

# ── Drawing helpers ────────────────────────────────────────
def to_pixels(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    lm = np.array([[p.x * w, p.y * h] for p in hand_landmarks])
    return lm.astype(int)

def draw_hand(frame, coords, accent_color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(coords[a]), tuple(coords[b]), accent_color, 2)
    for i, (x, y) in enumerate(coords):
        is_tip  = i in FINGER_TIPS
        color   = accent_color if is_tip else (240, 240, 240)
        radius  = 9 if is_tip else 5
        cv2.circle(frame, (x, y), radius, color, cv2.FILLED)

def draw_label_box(frame, text, color, coords, hand_idx):
    """Draw a rounded label near the wrist of each hand."""
    wrist = coords[0]
    x = max(wrist[0] - 10, 10)
    y = max(wrist[1] + 55, 55)

    # Background pill
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    pad = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Colored accent line above text
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y - th - 4),
                  color, -1)

    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

def draw_legend(frame):
    """Small gesture reference panel in top-right corner."""
    h, w = frame.shape[:2]
    entries = [v for k, v in GESTURE_INFO.items() if k not in ("None", "Unknown")]
    panel_w, line_h = 230, 26
    panel_h = len(entries) * line_h + 20
    x0, y0 = w - panel_w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "Gestures", (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for i, (label, color) in enumerate(entries):
        y = y0 + 20 + (i + 1) * line_h
        cv2.circle(frame, (x0 + 12, y - 5), 5, color, -1)
        cv2.putText(frame, label, (x0 + 24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

# ── Main loop ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

timestamp_ms = 0
prev_time    = time.time()

print("Gesture recognizer running — press Q to quit")
print("Supported: Closed_Fist | Open_Palm | Pointing_Up | Thumb_Down | Thumb_Up | Victory | ILoveYou")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 1
    recognizer.recognize_async(mp_image, timestamp_ms)

    result = latest["result"]
    if result and result.hand_landmarks:
        for idx, hand_lms in enumerate(result.hand_landmarks):

            # ── Get gesture name + confidence ──
            gesture_name = "None"
            confidence   = 0.0
            if result.gestures and idx < len(result.gestures):
                top = result.gestures[idx][0]
                gesture_name = top.category_name
                confidence   = top.score

            label_text, accent_color = GESTURE_INFO.get(
                gesture_name, (gesture_name, (200, 200, 200))
            )

            # ── Draw skeleton in gesture color ──
            coords = to_pixels(hand_lms, frame.shape)
            draw_hand(frame, coords, accent_color)

            # ── Draw gesture label near wrist ──
            handedness = result.handedness[idx][0].display_name
            display    = f"{label_text}  {confidence:.0%}"
            draw_label_box(frame, display, accent_color, coords, idx)

    # ── Legend + FPS ───────────────────────────────────────
    draw_legend(frame)

    now       = time.time()
    fps       = 1 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("MediaPipe Gesture Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()