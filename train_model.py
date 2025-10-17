# train_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Load prepared data
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')

print("Shapes:", X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

seq_length = X_train.shape[1]
num_features = X_train.shape[2]
num_classes = Y_train.shape[1]

# Build model: Conv1D -> Conv1D -> (optional) BiLSTM -> Dense
def build_model(use_lstm=True):
    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', padding='same', input_shape=(seq_length, num_features)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    if use_lstm:
        # now a small Bidirectional LSTM to capture temporal dynamics
        model.add(Bidirectional(LSTM(128, return_sequences=False)))
    else:
        model.add(GlobalAveragePooling1D())

    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = build_model(use_lstm=True)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint('models/gesture_model_conv1d_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

# class weights
y_int = np.argmax(Y_train, axis=1)
cw = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=80,
    batch_size=16,
    callbacks=[checkpoint, reduce_lr, early_stop],
    class_weight=class_weights,
    verbose=2
)

model.save('models/gesture_model_conv1d_last.h5')
print("Training complete. Best model saved to models/gesture_model_conv1d_best.h5")
