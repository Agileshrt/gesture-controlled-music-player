# real_time_gesture.py
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import pygame
import pyautogui
import os

# ----------------------------
# Parameters
# ----------------------------
MODEL_PATH = "models/gesture_model_conv1d_best.h5"
LABELS = list(np.load('label_map.npy'))  # gesture labels
SEQ_LENGTH = 30
CONF_THRESHOLD = 0.85
SMOOTHING_WINDOW = 7

SONG_FOLDER = "songs"
song_list = [os.path.join(SONG_FOLDER, f) for f in os.listdir(SONG_FOLDER) if f.endswith(".mp3")]
song_index = 0

# ----------------------------
# Load Model
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# Mediapipe Hands Setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# ----------------------------
# Buffers
# ----------------------------
seq = []
pred_buf = deque(maxlen=SMOOTHING_WINDOW)
last_time = 0
gesture_locked = False

# ----------------------------
# Initialize Pygame Mixer
# ----------------------------
pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)
is_playing = False

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_sequence_live(seq_list):
    arr = np.array(seq_list, dtype=np.float32).reshape(SEQ_LENGTH, 21, 3)
    root = arr[:, 0:1, :]
    arr_rel = arr - root
    max_xy = np.max(np.abs(arr_rel[:, :, 0:2]))
    if max_xy < 1e-6:
        max_xy = 1.0
    arr_rel[:, :, 0:2] /= max_xy
    arr_rel[:, :, 2] /= (max_xy + 1e-6)
    arr_flat = arr_rel.reshape(SEQ_LENGTH, 63)
    mu = arr_flat.mean()
    std = arr_flat.std()
    arr_flat = (arr_flat - mu) / (std + 1e-6)
    return arr_flat.reshape(1, SEQ_LENGTH, 63)

def perform_action(gesture):
    global song_index, is_playing
    try:
        # Discrete gestures: play/pause/next/previous
        if gesture == 'play':
            if not is_playing:
                pygame.mixer.music.load(song_list[song_index])
                pygame.mixer.music.play()
                is_playing = True
            else:
                pygame.mixer.music.unpause()
        elif gesture == 'pause':
            pygame.mixer.music.pause()
        elif gesture == 'next':
            song_index = (song_index + 1) % len(song_list)
            pygame.mixer.music.load(song_list[song_index])
            pygame.mixer.music.play()
            is_playing = True
        elif gesture == 'previous':
            song_index = (song_index - 1) % len(song_list)
            pygame.mixer.music.load(song_list[song_index])
            pygame.mixer.music.play()
            is_playing = True
        # Continuous gestures: volume up/down
        elif gesture == 'volume_up':
            pyautogui.press('volumeup')
        elif gesture == 'volume_down':
            pyautogui.press('volumedown')
    except Exception as e:
        print("Error performing action:", e)

# ----------------------------
# Video Capture
# ----------------------------
cap = cv2.VideoCapture(0)
print("🎶 Hand Gesture Music Control (Press ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # ----------------------------
    # Collect landmarks
    # ----------------------------
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
            seq.append(landmarks)
    else:
        seq = []
        gesture_locked = False

    if len(seq) > SEQ_LENGTH:
        seq.pop(0)

    # ----------------------------
    # Predict Gesture
    # ----------------------------
    gesture = "Collecting..."
    conf = 0.0

    if len(seq) == SEQ_LENGTH:
        x = preprocess_sequence_live(seq)
        y = model.predict(x, verbose=0)[0]
        pred_buf.append(y)
        avg = np.mean(pred_buf, axis=0)
        gid = np.argmax(avg)
        conf = np.max(avg)
        gesture = LABELS[gid]

        # ----------------------------
        # Continuous volume gestures
        if gesture in ['volume_up', 'volume_down'] and conf > CONF_THRESHOLD:
            perform_action(gesture)

        # Discrete gestures: trigger once per occurrence
        elif gesture in ['play', 'pause', 'next', 'previous'] and conf > CONF_THRESHOLD:
            if not gesture_locked:
                perform_action(gesture)
                gesture_locked = True
        # Unlock when gesture changes or confidence drops
        if gesture_locked and (conf < CONF_THRESHOLD or gid != np.argmax(avg)):
            gesture_locked = False

    # ----------------------------
    # Display
    # ----------------------------
    cv2.putText(frame, f"{gesture} ({conf:.2f})", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Gesture Music Player", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
