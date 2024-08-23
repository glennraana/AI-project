
#Tracking People in a Zone with YOLOv8

I’ve developed a program that detects and counts people within a defined zone using a live webcam stream. The counter updates as people enter or leave the zone. This solution leverages YOLOv8 as a pre-trained model and works very effectively.

#Project Overview
Initially, I created the solution in Google Colab, utilizing the power of a T4 GPU. The process was straightforward, thanks to the available documentation.

However, I wanted to implement the solution in VS Code to use it with my webcam locally. This transition presented more challenges, particularly concerning the installation of dependencies and ensuring all files were correctly placed. I encountered several issues with getting Detectron2 to work, but I found the solution in their GitHub documentation.

#Steps to Recreate the Solution
Set Up a Clean Virtual Environment: The first step was to create a completely clean virtual environment to avoid conflicts.
Use Jupyter Notebook in VS Code: I installed all the necessary requirements and downloaded the models using a Jupyter Notebook in VS Code. This allowed me to run code line by line and maintain control over the process.
Testing with Smaller Scripts: After everything was installed, I created three smaller scripts to test the setup.

#Script 1: take_detection.py
This script was used to test the model and ensure it detected objects as expected. It can use either images or a video stream to make predictions on a frame. This is where you can see YOLO in action, detecting multiple objects in a test image.
#Script 2: detection_just_people.py
I refined the code to detect only people in the room. This script is an updated version that focuses solely on detecting individuals, and it works well for this purpose.
#Script 3: detection_with_zone_live.py

The main goal of the project was to create a counter for people within a defined polygonal zone. I split my living room into two zones (50/50) and defined a zone where people could enter and exit. I wanted this to work live with my webcam rather than using an uploaded video. I used OpenCV to open the webcam and defined the zone using a polygon based on the dimensions of my camera's frame. After some research and with the help of ChatGPT, I resolved issues with the people counter. The solution now works, although it’s not perfect and could be refined in future projects.


