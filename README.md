
# OpenPose Implementation

This repository contains an implementation of OpenPose for human pose detection and estimation. The project demonstrates how to use OpenPose to detect and visualize keypoints in images and videos, providing insights into body posture and movement.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

OpenPose is a popular library for real-time multi-person keypoint detection. This project aims to implement and analyze OpenPose for detecting human body keypoints in static images and videos. The implementation includes experimentation with different confidence thresholds to understand their impact on detection accuracy.

## Features

- Detects human body keypoints in images and videos.
- Visualizes keypoints and skeletal structure.
- Analyzes the effect of confidence thresholds on detection accuracy.
- Supports real-time pose estimation in videos.

## Setup

### Prerequisites

- Python 3.x
- OpenCV
- OpenPose model files

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rktim/Open_pose.git
   cd Open_pose
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the OpenPose model files and place them in the project directory.

## Usage

### Image Pose Detection

To detect poses in a static image, use the following command:
```bash
python op.py --image path/to/image.jpg
```

### Video Pose Detection

To detect poses in a video, use the following command:
```bash
python openpose.py --video path/to/video.mp4
```

### Adjusting Confidence Threshold

You can adjust the confidence threshold by modifying the `thr` variable in the `op.py` script. Experiment with different thresholds to observe their impact on detection accuracy.

## Results

The project includes sample results demonstrating pose detection on images and videos with varying confidence thresholds. The results highlight the trade-off between precision and recall when adjusting the confidence threshold.
![image](https://github.com/user-attachments/assets/aa95096c-a3c0-48cb-8f02-973e1aea624b)


## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch-name`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides a clear and concise overview of your OpenPose project, making it easier for others to understand, set up, and contribute to your work.
