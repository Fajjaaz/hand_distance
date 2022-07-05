# hand_distance

Make sure to install the requirements, this can be done in cmd by running the following code: pip install -r requirements.txt

For this code, you need to connect a intel Realsense Depth Camera D455 to your computer. The hand_distance_cleaned.py script will open a screen that shows both the color and depth image from the camera. Using RANSAC, the code will determine the distance of a small rectangle around the wrist. If one wants to use different landmarks from the hand, these can be changed in the code. The default is set to the wrist.

In the hand_distance.py script, the uncleaned version of the code can be found. This script can also be used by people that want to further experiment with the code and the depth camera. The values for the depth are being saved in the values_depth.csv file and can be used for further analysis.

The script can be executed by running the follow line from command line: python hand_distance_cleaned.py
