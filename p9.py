import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition

# 1. SETUP: Create a directory named 'ImagesAttendance' in the same folder as this script.
# 2. Put images of the people you want to recognize in that directory.
# 3. Name the image files exactly as the person's name (e.g., 'Akil.jpg', 'Elon Musk.png').

path = 'ImagesAttendance'
images = []
classNames = []

# Create directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created.")
    print(f"Please place your reference images inside the '{path}' folder and run again.")
    print("Example: 'ImagesAttendance/John Doe.jpg'")

myList = os.listdir(path) if os.path.exists(path) else []

# Load images and extract class names (which are the file names without extensions)
for cl in myList:
    # Ignore hidden files like .DS_Store
    if cl.startswith('.'):
        continue
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print("Loaded classes:", classNames)

# Function to compute encodings for all known images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            # Detect the face encoding
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            # If no face is found in the image, skip it
            print("Warning: No face found in one of the images.")
    return encodeList

def markAttendance(name):
    """
    Logs the attendance to a CSV file.
    If the name is not yet present for today, it adds a new record.
    """
    filename = 'Attendance.csv'
    
    # Create file with headers if it doesn't exist
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('Name,Time,Date')

    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dString = now.strftime('%Y-%m-%d')
            f.writelines(f'\n{name},{dtString},{dString}')

# Prepare known face encodings
print("Encoding reference images... This might take a moment.")
encodeListKnown = findEncodings(images)
print(f"Encoding Complete. {len(encodeListKnown)} encodings found.")

if len(encodeListKnown) == 0:
    print("No faces encoded. Please add images to the 'ImagesAttendance' folder and restart.")
else:
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to access Webcam")
            break
            
        # Scale down the image to 1/4 size for faster processing during face recognition
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # Convert BGR (OpenCV) to RGB (face_recognition)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Find all the faces and their encodings in the current frame
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            # Compare current face encoding to known encodings
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
            # Find the best match index
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < 0.50: # Added a threshold for better accuracy
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = "UNKNOWN"

            # The face locations from imgS are scaled down by 1/4, so we multiply by 4 to get original coordinates
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw a bounding box around the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255), 2)
            # Draw a filled rectangle for the label text
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255), cv2.FILLED)
            # Add the name label
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Webcam Face Recognition', img)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
