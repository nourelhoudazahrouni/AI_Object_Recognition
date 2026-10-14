# AI Object Recognition Camera

A real-time object detection application built with Python, OpenCV, and YOLO-World. The application uses a webcam to detect selected objects, display detection labels, count detected objects, measure FPS, and capture screenshots.

## Features

- **Real-time object detection** using a webcam.
- **Custom object categories** configured with YOLO-World.
- **Bounding boxes and labels** displayed around detected objects.
- **Object counting** showing the total number of detections.
- **Individual class counts** showing how many objects of each category are detected.
- **FPS monitoring** to display the approximate processing speed.
- **Screenshot capture** with timestamped filenames.
- **Keyboard controls** for capturing images and closing the application.

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO-World
- Counter from Python's `collections` module
- `datetime` for screenshot filenames

## Requirements

- Windows, Linux, or macOS
- Python 3.10 or a compatible Python version supported by the installed dependencies
- A webcam
- An internet connection for downloading model weights when needed

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nourelhoudazahrouni/AI_Object_Recognition.git
cd AI_Object_Recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows using Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install the dependencies

```bash
python -m pip install opencv-python ultralytics
```

### 4. Run the application

```bash
python main.py
```

The application loads the YOLO-World model and opens the webcam. The model weights may be downloaded automatically on the first run.

## Keyboard Controls

| Key | Action |
|---|---|
| `S` | Capture a screenshot |
| `Q` | Quit the application |

Screenshots are saved in the current project directory with timestamped filenames, for example:

`capture_2026-10-14_16-05-10.jpg`

## Configured Object Categories

The application is configured to look for:

- Pen
- Banknote
- Paper currency
- Cell phone
- Toothbrush
- Person
- Bottle
- Book
- Cup
- Laptop

The model's predictions depend on the input image, lighting, object orientation, and model capabilities.

## Limitations

This project uses a pretrained open-vocabulary object detection model. It can misclassify objects or fail to detect them, particularly when objects are small, partially hidden, or visually similar.

Banknote detection does not verify whether a banknote is genuine or determine its denomination.

## Future Improvements

- Improve recognition accuracy for specific objects.
- Add a graphical user interface.
- Save detection history and statistics.
- Support additional object categories.
- Evaluate performance under different lighting conditions.
- **Detection history (CSV):** Automatically records the date, time, total number of detected objects, and object counts approximately every two seconds.
- **Data analysis:** The CSV history can be opened in Microsoft Excel or another spreadsheet application for further analysis.

## Detection History

The application creates a `detection_history.csv` file when it runs. Each record contains:

| Column | Description |
|---|---|
| Date | Date of the recorded detection |
| Time | Time of the recorded detection |
| Total Objects | Total number of detected objects |
| Object Counts | Counts grouped by predicted object category |

The history contains model predictions and may include incorrect classifications. The CSV file is generated locally and is not included in the Git repository by default.

## Author

**Nour El Houda Zahrouni**

GitHub: [nourelhoudazahrouni](https://github.com/nourelhoudazahrouni)
