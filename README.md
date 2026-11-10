# Sight Assist: Real-Time Object Identification and Assistance for the Blind

Sight Assist is a real-time assistive vision system designed to help visually impaired users understand and navigate their surroundings. The system uses computer vision and deep learning to detect objects from live camera input, estimate their proximity, prioritize important obstacles, and provide audio feedback through text-to-speech.

## Features

- Real-time object detection using YOLO-based deep learning
- Live camera/video processing
- Object classification using the COCO dataset
- Monocular distance estimation
- Priority-based obstacle filtering
- Text-to-speech audio feedback
- Continuous real-time assistance
- Low-latency on-device processing

## System Workflow

```text
Camera Input
     ↓
Frame Preprocessing
     ↓
Object Detection
     ↓
Distance Estimation
     ↓
Priority Filtering
     ↓
Text-to-Speech
     ↓
Audio Feedback
```

## Technologies Used

- **Programming:** Python
- **Deep Learning:** YOLO, CNN
- **Computer Vision:** OpenCV
- **Deep Learning Frameworks:** PyTorch / TensorFlow
- **Numerical Processing:** NumPy
- **Dataset:** COCO
- **Audio:** Google Text-to-Speech (gTTS)

## How It Works

The system continuously captures video frames through a camera. Each frame is preprocessed and passed to the object detection model to identify objects in the environment.

The detected objects are then analyzed using their bounding-box information to estimate their approximate distance from the user. Objects are prioritized based on their relevance and proximity, allowing critical obstacles to be communicated first.

The processed information is converted into speech using a text-to-speech module and delivered to the user through speakers or earphones.

## Performance

During controlled testing, the system demonstrated:

- Approximately **200 ms average video frame processing time**
- **96% object detection success rate** in 100 test trials
- Positive user feedback regarding audio clarity, responsiveness, and navigation assistance

## Hardware

The prototype was developed using:

- HD RGB camera
- Laptop/processing unit
- Earphones or speakers
- Portable power supply

## Project Architecture

The system follows a modular architecture consisting of:

1. Camera Module
2. Frame Preprocessing
3. Object Detection
4. Distance Estimation
5. Priority Filtering
6. Text-to-Speech
7. Audio Output
8. Continuous Assistance Loop

## Future Enhancements

- GPS-based navigation
- Wearable smart-glasses integration
- Improved depth estimation using depth cameras or LiDAR
- Multilingual voice assistance
- Scene understanding
- Facial recognition
- Optimization for low-power and wearable devices

## Project Objective

The primary objective of Sight Assist is to improve independence, safety, and situational awareness for visually impaired users by converting visual information from the surrounding environment into meaningful audio feedback.

## Authors

**Harshini S.**  
Artificial Intelligence and Data Science  
Dr. Mahalingam College of Engineering and Technology
