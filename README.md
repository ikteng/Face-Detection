# Face-Detection
This program uses cv2 and matplotlib. It provides a simple example of face and eye detection using pre-trained Haar Cascade classifiers, which are part of the OpenCV library. The real-time face detection part allows users to see live face detection through their webcam.

There are 2 versions in this program:
1. face detection from images
2. face detection in real-time

## face_detection_from_image(image_path) function:
- Load Haar Cascade classifiers for face and eye detection from OpenCV.
- Read an image specified by the image_path.
- Convert the image to grayscale.
- Detect faces using the face cascade with adjustable parameters (scaleFactor, minNeighbors, and minSize).
- For each detected face, draw a blue rectangle around it and create a Region of Interest (ROI) for that face.
- Detect eyes within the face region using the eye cascade and draw green rectangles around them.
- Display the resulting image with rectangles around faces and eyes using matplotlib.

## real_time_face_detection() function:
- Load Haar Cascade classifiers for face and eye detection from OpenCV.
- Access the default webcam (you can change the parameter if multiple webcams are available).
- Continuously capture frames from the webcam.
- Convert each frame to grayscale for face detection.
- Detect faces in the frame using the face cascade with adjustable parameters.
- For each detected face, draw a blue rectangle around it and create an ROI for that face.
- Detect eyes within the face region using the eye cascade and draw green rectangles around them.
- Display the frame with rectangles around faces and eyes in real-time.
- The real-time face detection continues until the user presses 'q' or 'esc' to exit the program.
- Release the video capture and close all OpenCV windows.

## main Method:
- calls real_time_face_detection() to demonstrate real-time face detection from the webcam.
- There is a commented-out section (image_path="Face Detection\obamas.jpg" and face_detection_from_image(image_path)) that can be used to perform face detection on a single image by specifying the image file path.

# Things to Improve
1. emotion, gender and age predictions
2. facial recognition
