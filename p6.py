import cv2
import numpy as np

# Define model paths
face_proto = "deploy.prototxt"
face_model = "res10_300x300_ssd_iter_140000.caffemodel"
age_proto = "deploy_age.prototxt"
age_model = "age_net.caffemodel"
gender_proto = "deploy_gender.prototxt"
gender_model = "gender_net.caffemodel"

# Define age and gender lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load DNN networks
try:
    face_net = cv2.dnn.readNet(face_model, face_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
except cv2.error as e:
    print("Error loading models. Please ensure the caffe models and prototxt files are correctly downloaded in the directory.")
    print("Exception:", e)
    exit(1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get face bounding boxes
    h, w = frame.shape[:2]
    # Blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold for face detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Validate box coordinates within frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Extract face Region of Interest
            face = frame[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20: # Ignore trivially small/invalid boxes
                continue

            # Blob for Age & Gender prediction
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict Gender
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict Age
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            y_label = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age & Gender Prediction", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
