
import cv2
from ultralytics import YOLO

print("Loading AI model...")

model = YOLO("yolo11s.pt")

print("AI model loaded!")
print("Starting webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("Object detection started. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read the webcam.")
            break

        results = model(frame, conf=0.6)

        annotated_frame = results[0].plot()

        cv2.imshow("AI Object Recognition Camera", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
