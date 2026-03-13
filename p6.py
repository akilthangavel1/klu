import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Analyze face
        result = DeepFace.analyze(
            frame,
            actions=['age', 'gender'],
            enforce_detection=False
        )

        age = result[0]['age']
        gender = result[0]['dominant_gender']

        text = f"{gender}, Age: {age}"

        cv2.putText(frame, text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    except:
        pass

    cv2.imshow("Age & Gender Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()