
import cv2
import time
import csv
import os
from ultralytics import YOLOWorld
from collections import Counter
from datetime import datetime

print("Loading AI model...")
model = YOLOWorld("yolov8s-world.pt")

model.set_classes([
    "pen",
    "banknote",
    "paper currency",
    "cell phone",
    "toothbrush",
    "person",
    "bottle",
    "book",
    "cup",
    "laptop"
])

history_file = "detection_history.csv"

if not os.path.exists(history_file):
    with open(history_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date",
            "Time",
            "Total Objects",
            "Object Counts"
        ])

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("AI Object Recognition Camera started!")
    print("Press S to take a screenshot.")
    print("Press Q to quit.")

    previous_time = time.time()
    last_log_time = 0

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read the webcam.")
            break

        results = model(frame, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()

        object_count = len(results[0].boxes)

        class_names = results[0].names
        detected_classes = results[0].boxes.cls.tolist()

        class_counts = Counter(
            class_names[int(class_id)]
            for class_id in detected_classes
        )

        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - previous_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        previous_time = current_time

        # Save detection history every 2 seconds
        current_log_time = time.time()

        if current_log_time - last_log_time >= 2:
            with open(
                history_file, "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d"),
                    datetime.now().strftime("%H:%M:%S"),
                    object_count,
                    dict(class_counts)
                ])

            last_log_time = current_log_time

        # Display total detected objects
        cv2.putText(
            annotated_frame,
            f"Objects detected: {object_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display count for each object category
        y_position = 60

        for name, count in class_counts.items():
            cv2.putText(
                annotated_frame,
                f"{name}: {count}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_position += 30

        # Display FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, y_position + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("AI Object Recognition Camera", annotated_frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S_%f"
            )
            filename = f"capture_{timestamp}.jpg"

            saved = cv2.imwrite(filename, annotated_frame)

            if saved:
                print(f"Screenshot saved as {filename}")
            else:
                print("Error: Could not save screenshot.")

        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
