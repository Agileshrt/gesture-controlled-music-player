import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui
from collections import deque
import time

MODEL_PATH = "models/gesture_model_conv1d_best.h5"
model = load_model(MODEL_PATH)
LABELS = list(np.load('label_map.npy'))
SEQ_LENGTH = 30
CONF_THRESHOLD = 0.85
SMOOTHING_WINDOW = 7
COOLDOWN = 1.2  # seconds

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

seq = []
pred_buf = deque(maxlen=SMOOTHING_WINDOW)
last_time = 0

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
    mu = arr_flat.mean(); std = arr_flat.std()
    arr_flat = (arr_flat - mu) / (std + 1e-6)
    return arr_flat.reshape(1, SEQ_LENGTH, 63)

def perform_action(gesture):
    try:
        if gesture == 'play' or gesture == 'pause':
            pyautogui.press('playpause')
        elif gesture == 'next':
            pyautogui.press('nexttrack')
        elif gesture == 'previous':
            pyautogui.press('prevtrack')
        elif gesture == 'volume_up':
            pyautogui.press('volumeup')
        elif gesture == 'volume_down':
            pyautogui.press('volumedown')
    except Exception as e:
        print("pyautogui error:", e)

cap = cv2.VideoCapture(0)
print("🎶 Gesture control (ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
            seq.append(landmarks)

    if len(seq) > SEQ_LENGTH:
        seq.pop(0)

    if len(seq) == SEQ_LENGTH:
        x = preprocess_sequence_live(seq)
        y = model.predict(x, verbose=0)[0]
        pred_buf.append(y)
        avg = np.mean(pred_buf, axis=0)
        gid = np.argmax(avg)
        conf = np.max(avg)
        gesture = LABELS[gid]

        if time.time() - last_time > COOLDOWN and conf > CONF_THRESHOLD:
            perform_action(gesture)
            last_time = time.time()
            print("Performed:", gesture, "conf:", conf)

        # display
        cv2.putText(frame, f"{gesture} {conf:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No hand / collecting...", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Gesture Music Player", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
