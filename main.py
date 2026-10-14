import cv2
import time
from ultralytics import YOLO

print("Loading AI model...")
model = YOLO("yolo11s.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("AI Object Recognition Camera started!")
    print("Press S to take a screenshot.")
    print("Press Q to quit.")

    previous_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read the webcam.")
            break

        results = model(frame, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()

        object_count = len(results[0].boxes)

        current_time = time.time()
        elapsed_time = current_time - previous_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        previous_time = current_time

        cv2.putText(
            annotated_frame,
            f"Objects detected: {object_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("AI Object Recognition Camera", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite("capture.jpg", annotated_frame)
            print("Screenshot saved as capture.jpg")

        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
