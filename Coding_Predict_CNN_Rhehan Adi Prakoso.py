import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# ===============================
# 1. Load Model & Label
# ===============================
model = load_model('cnn_model.h5')

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

print("✅ Model & Label berhasil dimuat!")

# ===============================
# 2. Prediksi Realtime via Webcam
# ===============================
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mode Night Vision
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi
    pred = model.predict(img)
    idx = np.argmax(pred)
    label = class_labels[idx]
    confidence = pred[0][idx] * 100  # ubah ke persen

    # Tampilkan hasil di frame webcam
    text = f'Class: {label} | Accuracy: {confidence:.2f}%'
    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
