import cv2
import numpy as np

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the lower and upper boundaries of a "blue"
    # object in the HSV color space.
    # Note: OpenCV uses H: 0-179, S: 0-255, V: 0-255
    # You can change these values to track different colors:
    # Green: lower (~40, 50, 50), upper (~80, 255, 255)
    # Red: lower (~160, 50, 50), upper (~180, 255, 255) OR lower (~0, 50, 50), upper (~10, 255, 255)
    color_lower = np.array([100, 150, 0], dtype=np.uint8)
    color_upper = np.array([140, 255, 255], dtype=np.uint8)

    print("Tracking blue color. Press 'q' to quit.")

    while True:
        # Read the next frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Blurred frame helps in reducing noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # Convert the frame to HSV (Hue, Saturation, Value) color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Construct a mask for the color, then perform a series of
        # dilations and erosions to remove any small blobs left in the mask
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None

        # Only proceed if at least one contour was found
        if len(contours) > 0:
            # Find the largest contour in the mask, then use it to compute
            # the minimum enclosing circle and centroid
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Calculate the center (Moments)
            M = cv2.moments(c)
            # Avoid division by zero
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = (int(x), int(y))

            # Only proceed if the radius meets a minimum size
            # This filters out tiny specks of the color
            if radius > 10:
                # Draw the circle and centroid on the frame
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Add text to display the current tracking
                cv2.putText(frame, "Tracking Object", (int(x) - int(radius), int(y) - int(radius) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show the output frame
        cv2.imshow("Frame", frame)
        # Uncomment the next line to show the mask window as well
        # cv2.imshow("Mask", mask) 

        # If the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Cleanup the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
