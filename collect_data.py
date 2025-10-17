import cv2
import mediapipe as mp
import numpy as np
import os

# -------------------------------
# 1. Gesture Names (edit here)
# -------------------------------
actions = ['play', 'pause', 'next', 'previous', 'volume_up', 'volume_down']

# -------------------------------
# 2. Settings
# -------------------------------
seq_length = 30   # number of frames per sequence
num_sequences = 50  # how many sequences you want to collect per gesture

# Create dataset folder
os.makedirs('dataset', exist_ok=True)

# Mediapipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# 3. Data Collection Loop
# -------------------------------
cap = cv2.VideoCapture(0)

for action in actions:
    action_dir = os.path.join('dataset', action)
    os.makedirs(action_dir, exist_ok=True)
    print(f"\n📸 Collecting data for gesture: '{action}'")

    for seq_num in range(num_sequences):
        print(f"  ▶ Sequence {seq_num+1}/{num_sequences} ...")
        data = []

        # Show countdown before each sequence (for you to get ready)
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Start in {countdown}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow('Collecting Gesture', frame)
            cv2.waitKey(1000)

        # Collect 30 frames per sequence
        while len(data) < seq_length:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    landmark = []
                    for lm in handLms.landmark:
                        landmark.extend([lm.x, lm.y, lm.z])
                    data.append(landmark)

            # Show live feed
            cv2.putText(frame, f"{action} | Frame: {len(data)}/{seq_length}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collecting Gesture', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save the collected sequence
        if len(data) == seq_length:
            np.save(os.path.join(action_dir, f"{seq_num}.npy"), np.array(data))
            print(f"    ✅ Saved sequence {seq_num+1}")

print("\n🎉 All gestures collected successfully!")
cap.release()
cv2.destroyAllWindows()
