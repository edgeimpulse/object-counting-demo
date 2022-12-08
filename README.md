# Object Counting Demo
A quick demo to show one way to count object passing through a FOMO window. Objects are counted as they leave the top of the frame.

The scripts provided scan a slice of the top of the window over a number of frames to detect when inferencing goes from 1 to 0 (similar to detecting a button press). To debounce, an object is only counted if there are N number of 0 values detected after a 1 is.

An example object counting classifier file is provided here (see .eim) from this project: https://studio.edgeimpulse.com/public/165829/latest 

To run live with a webcam use:
`python classify-camera.py <path to classifier file.eim> <camera port>`

To run on an existing video: `python classify-video.py <path to classifier file.eim> <path to .mp4>`

Examples:
