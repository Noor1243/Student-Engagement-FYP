# Student Engagement Monitoring System

A real-time engagement detection tool designed to monitor and analyze student attentiveness in virtual classrooms and recorded videos using advanced facial landmark-based features. This system leverages computer vision techniques, including RetinaFace for face detection, to provide actionable insights into student focus, helping educators improve teaching effectiveness and participation.

---

## Built With

- **Python**
- OpenCV
- MediaPipe
- NumPy
- RetinaFace (for multi-face detection)
- Matplotlib (visualizations)

---

## Project Overview

This system processes Zoom-style grid videos containing multiple students, detecting and tracking each face individually using RetinaFace for precise face localization. It extracts detailed facial landmarks via MediaPipe FaceMesh to compute engagement-related metrics such as:

- Head pose (pitch, yaw, roll)
- Eye contact and gaze estimation
- Blinking and yawning detection via mouth aspect ratio
- Distraction duration and focus classification

The project applies a landmark-based Support Vector Machine (SVM) model to classify whether each student is **Focused** or **Distracted** in real time. Engagement metrics are visualized on video frames and summarized via histograms and textual reports.

---

## Target Use Case: Education (E-Learning Platforms)

**Problem:**  
Educators face challenges in monitoring student engagement during virtual classes, making it difficult to adapt teaching strategies effectively.

**Solution:**  
This tool analyzes student video feeds to provide real-time feedback on attentiveness, enabling timely intervention by teachers.

**Output:**  
- Visual overlays marking each student’s engagement state during live or recorded sessions  
- Per-student engagement statistics and trends  
- Engagement summaries for post-class analysis

---

## How It Works

### Multiple Face Detection & Tracking

- Uses **RetinaFace** for robust multi-face detection in each video frame.
- Applies **MediaPipe FaceMesh** to extract 3D facial landmarks for each detected face.
- Assigns a unique ID to each student by tracking nose tip coordinates, maintaining consistent identification across frames.
- Extracts engagement features per student using custom logic.

### Feature Extraction & Engagement Classification

- Computes eye distance, mouth openness, and head tilt from landmarks.
- Extracts behavioral features such as blinking and gaze direction.
- Classifies focus state using a **Support Vector Machine (SVM)** trained on landmark features.

### Output & Visualization

- Annotates live video frames with student ID, focus status, and head pose.
- Saves processed videos with overlays.
- Generates histograms summarizing engagement distributions per student.
- Exports textual summary reports showing focused frame ratios.

---

## Key Facial Landmarks Utilized

- Nose tip for head pose and tracking  
- Eyes and iris landmarks for gaze and blink detection  
- Lips for yawning detection and mouth aspect ratio

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy
- RetinaFace (Python package)
- Matplotlib

### Installation

1. Clone the repository:


git clone https://github.com/yourusername/student-engagement-monitoring.git
cd student-engagement-monitoring
pip install -r requirements.txt
python app.py

### Usage
The script processes the input video, detects and tracks all student faces using RetinaFace, extracts engagement features, and labels each student as Focused or Distracted per frame.

Output includes:

Processed video with overlay annotations (output/processed_video.mp4)

Engagement histograms per student (output/student_<id>_hist.png)

Engagement summary text file (output/engagement_summary.txt)

Contributing
Contributions are welcome! Please fork the repo, create your feature branch, commit your changes, and submit a pull request.

License
This project is licensed under the MIT License.
