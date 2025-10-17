# evaluate_model.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

actions = list(np.load('label_map.npy'))
X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')
model = load_model('models/gesture_model_conv1d_best.h5')

preds = model.predict(X_val, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(Y_val, axis=1)

print(classification_report(y_true, y_pred, target_names=actions))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
