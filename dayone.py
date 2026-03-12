import cv2
import os
from ultralytics import YOLO

video_path = "/content/video.mp4"

if not os.path.exists(video_path):
    print("No video has given as input")

else:

    print("Video found. Starting object detection...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(frame)


        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                label = object_name + " - the object was detected"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                cv2.putText(frame,
                            label,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)


        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Object detection completed.")
