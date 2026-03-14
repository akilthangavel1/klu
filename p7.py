import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model and prototxt
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

# Define the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Generate random colors for each class to draw bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the network from Caffe model
print("[INFO] Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
except cv2.error as e:
    print("Error loading models. Ensure that the caffemodel and prototxt files are downloaded.")
    print("Exception:", e)
    exit(1)

# Start the webcam stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the dimensions of the frame
    (h, w) = frame.shape[:2]
    
    # Preprocess the frame to create a blob for the neural network
    # MobileNet SSD expects 300x300 images with a scale factor of 0.007843 and mean [127.5, 127.5, 127.5]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    
    # Pass the blob through the network to get detections
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections (confidence < 50%)
        if confidence > 0.5:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])
            
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Prevent bounding boxes from going outside the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            
            # Draw the bounding box and label on the frame
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            
            # Put the label text properly
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[idx], 2)

    # Display the current frame
    cv2.imshow("Real-Time Object Detection", frame)
    
    # Press 'q' key to stop the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
