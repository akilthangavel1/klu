import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_classifier.empty():
    raise RuntimeError("Unable to load face detection model.")

model_path = os.path.join(os.path.dirname(__file__), "emotion_model.hdf5")
classifier = load_model(model_path, compile=False) if os.path.exists(model_path) else None

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to access the camera.")

while True:

    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if classifier is None:
        cv2.putText(
            frame,
            "emotion_model.hdf5 not found",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if classifier is None:
            continue
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))

        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))

        preds = classifier.predict(roi, verbose=0)
        label = emotion_labels[int(preds.argmax())]

        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Emotion Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
