# Face-and-Hand-Recognition

# Real-time Face and Hand Detection

This project implements a real-time face and hand detection system using OpenCV and MediaPipe. It leverages Haar cascades for efficient face, eye, and smile detection, and MediaPipe for robust hand tracking and finger counting.

## Features

* **Face Detection:** Detects faces in real-time using the pre-trained Haar cascade classifier.
* **Eye Detection and Status:** Identifies eyes within detected faces and provides status (Open/Closed) based on the number of eyes detected.
* **Smile Detection:** Detects smiles within detected faces.
* **Hand Detection:** Tracks hands in real-time using MediaPipe.
* **Finger Counting:** Counts the number of extended fingers for detected hands, providing gesture recognition (e.g., "One," "Two," "Five").
* **Real-time Visualization:** Displays the video feed with bounding boxes, eye/smile status, and finger count overlays.

## Technologies Used

* **Python:** The core programming language for the application.
* **OpenCV (`cv2`):** A powerful library for computer vision tasks, used for video capture, image processing, and Haar cascade integration.
* **MediaPipe (`mp`):** A framework for building machine learning pipelines, specifically used here for hand detection and landmark estimation.
* **Haar Cascades:** Pre-trained XML classifiers used for detecting faces, eyes, and smiles. The following cascade files are used:
    * `haarcascade_frontalface_default.xml`
    * `haarcascade_eye.xml`
    * `haarcascade_smile.xml`

## Setup and Installation

To run this project, you need to have Python installed, along with the necessary libraries.

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies:**

    Make sure you have `opencv-python` and `mediapipe` installed. If not, you can install them using pip:

    ```bash
    pip install opencv-python mediapipe
    ```

3.  **Place Haar Cascade XML files:**

    Ensure that the Haar cascade XML files (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`, `haarcascade_smile.xml`) are present in the same directory as your `Code.py` file, or in the OpenCV data path. Your code explicitly references them using `cv2.data.haarcascades`, so they should be available there. If you downloaded them separately, you might need to place them in the correct `site-packages/cv2/data` directory of your Python environment or in your project root.

## How to Run

1.  **Execute the Python script:**

    ```bash
    python Code.py
    ```

2.  A window titled "Face & Hand Detection" will open, displaying the real-time video feed from your webcam with the detections.

3.  To exit the application, press any key.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]
