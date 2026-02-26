import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import os
import pyautogui

MODEL_PATH = "models/gesture_recognizer.task"

pyautogui.FAILSAFE = True

# ── MediaPipe setup ─────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
GestureRecognizer     = mp.tasks.vision.GestureRecognizer
GestureRecognizerOpts = mp.tasks.vision.GestureRecognizerOptions
RunningMode           = mp.tasks.vision.RunningMode

# ── Gestures Map ────────────────────────────────────────────
GESTURE_INFO = {
    "None":        ("Idle",            (150, 150, 150)),
    "Unknown":     ("Unrecognized",    (110, 110, 110)),
    "Closed_Fist": ("Click",           (255, 120, 0)),
    "Open_Palm":   ("Open Hand",       (0, 200, 120)),
    "Pointing_Up": ("Move Cursor",     (0, 220, 255)),
    "Thumb_Down":  ("Vol -",           (0, 80, 220)),
    "Thumb_Up":    ("Vol +",           (0, 255, 150)),
    "Victory":     ("Alt Tab",         (200, 0, 255)),
    "ILoveYou":    ("Exit",            (255, 60, 120)),
}

# ── Hand skeleton ───────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
FINGER_TIPS = {4, 8, 12, 16, 20}

# ── Async result store ──────────────────────────────────────
latest = {"result": None}

def on_result(result, output_image, timestamp_ms):
    latest["result"] = result

# ── Build recognizer ────────────────────────────────────────
options = GestureRecognizerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result
)
recognizer = GestureRecognizer.create_from_options(options)

# ── Drawing helpers ─────────────────────────────────────────
def to_pixels(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    lm = np.array([[p.x * w, p.y * h] for p in hand_landmarks])
    return lm.astype(int)

def draw_hand(frame, coords, accent_color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(coords[a]), tuple(coords[b]), accent_color, 2)
    for i, (x, y) in enumerate(coords):
        radius = 8 if i in FINGER_TIPS else 5
        color  = accent_color if i in FINGER_TIPS else (230, 230, 230)
        cv2.circle(frame, (x, y), radius, color, cv2.FILLED)

def draw_label(frame, text, color, coords):
    x, y = coords[0]
    cv2.putText(frame, text, (x + 10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

# ── Gesture action logic ────────────────────────────────────
ACTION_COOLDOWN = 0.6
last_action_time = {}

def can_trigger(gesture):
    now = time.time()
    last = last_action_time.get(gesture, 0)
    if now - last > ACTION_COOLDOWN:
        last_action_time[gesture] = now
        return True
    return False

def handle_gesture_action(gesture, coords, frame_shape):
    if not can_trigger(gesture):
        return

    if gesture == "Closed_Fist":
        pyautogui.click()

    elif gesture == "Pointing_Up":
        tip = coords[8]
        h, w = frame_shape[:2]
        screen_w, screen_h = pyautogui.size()

        x = np.interp(tip[0], [0, w], [0, screen_w])
        y = np.interp(tip[1], [0, h], [0, screen_h])

        pyautogui.moveTo(x, y, duration=0.05)

    elif gesture == "Thumb_Up":
        pyautogui.press("volumeup")

    elif gesture == "Thumb_Down":
        pyautogui.press("volumedown")

    elif gesture == "Victory":
        pyautogui.hotkey("alt", "tab")

    elif gesture == "ILoveYou":
        print("Exit gesture detected")
        os._exit(0)

# ── Main loop ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

timestamp_ms = 0
fps_buffer = deque(maxlen=30)
prev_time = time.time()

print("Gesture control active — ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 1
    recognizer.recognize_async(mp_image, timestamp_ms)

    result = latest["result"]
    if result and result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]

        gesture = "None"
        confidence = 0.0
        if result.gestures:
            top = result.gestures[0][0]
            gesture = top.category_name
            confidence = top.score

        label, color = GESTURE_INFO.get(gesture, (gesture, (200, 200, 200)))
        coords = to_pixels(hand_lms, frame.shape)

        draw_hand(frame, coords, color)
        draw_label(frame, f"{label} {confidence:.0%}", color, coords)
        handle_gesture_action(gesture, coords, frame.shape)

    now = time.time()
    fps_buffer.append(1 / (now - prev_time + 1e-9))
    prev_time = now
    fps = int(sum(fps_buffer) / len(fps_buffer))

    cv2.putText(frame, f"FPS: {fps}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("Gesture PC Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()