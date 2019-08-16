# AI_drone

This repository is a playground for creating AI software to control a tello drone.
The communication with the tello drone is done using code from https://github.com/damiafuentes/DJITelloPy.

To run this code connect to the tello drones wifi then run ai_drone.py.
Once the program is loaded use the following commands.

For manually controlling the drone:
T - take off
L - land
W - move up
S - move down
A - rotate anti-clockwise
D - rotate clockwise
up arrow - move forward
down arrow - move back
left arrow - move left
right arrow - move right

Different AI modes:
P - follow human

The follow human mode workes by detecting bodies and faces. For this I use tensorflow's object detection API and some pre-trained models from the model zoo. The drone rotates and moves up, down, towards, or away from the human. 
Currently the distance is infered from the size of the bounding boxes as pydnet was not giving accurate enough depths.
TODO: research and find better alternative monocular depth detection.
