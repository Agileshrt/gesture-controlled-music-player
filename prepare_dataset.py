# prepare_dataset.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --------------------
actions = ['play', 'pause', 'next', 'previous', 'volume_up', 'volume_down']
seq_length = 30
dataset_dir = 'dataset'
augment_per_seq = 3  # number of augmented copies per raw sequence
# --------------------

def augment_jitter(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    for i in range(2, seq.shape[1], 3):  # zero noise for z
        noise[:, i] = 0.0
    return seq + noise

def augment_scaling(seq, scale_range=(0.92, 1.08)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return seq * scale

def augment_shift(seq, shift_range=0.02):
    shift = np.random.uniform(-shift_range, shift_range, seq.shape)
    for i in range(2, seq.shape[1], 3):
        shift[:, i] = 0.0
    return seq + shift

def augment_rotate(seq, angle_deg=6):
    # small rotation around z for x,y coordinates
    theta = np.deg2rad(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    arr = seq.reshape(seq_length, 21, 3).copy()
    # rotate x,y (about origin after root subtraction should be applied by preprocess)
    xy = arr[:, :, 0:2]
    xy_rot = np.empty_like(xy)
    xy_rot[:, :, 0] = c * xy[:, :, 0] - s * xy[:, :, 1]
    xy_rot[:, :, 1] = s * xy[:, :, 0] + c * xy[:, :, 1]
    arr[:, :, 0:2] = xy_rot
    return arr.reshape(seq_length, 63)

def random_augment(seq):
    seq2 = seq.copy()
    if np.random.rand() < 0.6:
        seq2 = augment_jitter(seq2, sigma=0.015)
    if np.random.rand() < 0.5:
        seq2 = augment_scaling(seq2)
    if np.random.rand() < 0.45:
        seq2 = augment_shift(seq2)
    if np.random.rand() < 0.3:
        seq2 = augment_rotate(seq2)
    return seq2

def preprocess_sequence(seq):
    # seq shape (30,63)
    arr = seq.reshape(seq_length, 21, 3).astype(np.float32)
    root = arr[:, 0:1, :]           # wrist (21-landmark index 0)
    arr_rel = arr - root            # root-relative
    max_xy = np.max(np.abs(arr_rel[:, :, 0:2]))
    if max_xy < 1e-6:
        max_xy = 1.0
    arr_rel[:, :, 0:2] /= max_xy
    arr_rel[:, :, 2] /= (max_xy + 1e-6)
    flat = arr_rel.reshape(seq_length, 63)
    # per-sequence standardization
    mu, std = flat.mean(), flat.std()
    flat = (flat - mu) / (std + 1e-6)
    return flat

# Load files
X, Y = [], []
print("Loading dataset from:", dataset_dir)
for idx, action in enumerate(actions):
    folder = os.path.join(dataset_dir, action)
    if not os.path.isdir(folder):
        print(f"  WARNING: missing folder {folder} (skip)")
        continue
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    print(f"  {action}: {len(files)} files")
    for f in files:
        try:
            seq = np.load(os.path.join(folder, f))
        except Exception as e:
            print("   - failed load:", f, e)
            continue
        if seq.shape != (seq_length, 63):
            continue
        prep = preprocess_sequence(seq)
        X.append(prep)
        Y.append(idx)
        for _ in range(augment_per_seq):
            aug = random_augment(prep)
            # re-preprocess augmented (standardize again)
            aug = preprocess_sequence(aug)
            X.append(aug)
            Y.append(idx)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int32)
print("Total sequences after augmentation:", len(X))

# Train/test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

Y_train = to_categorical(Y_train, num_classes=len(actions))
Y_val = to_categorical(Y_val, num_classes=len(actions))

# Save
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('X_val.npy', X_val)
np.save('Y_val.npy', Y_val)
np.save('label_map.npy', np.array(actions))
print("Saved X_train, Y_train, X_val, Y_val, label_map.npy")
print("Shapes:", X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
