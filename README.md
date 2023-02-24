# Object Counting Demo

For an on-device object counting example see: https://github.com/edgeimpulse/conveyor-counting-data-synthesis-demo

A quick demo to show one way to count object passing through a FOMO window. Objects are counted as they leave the top of the frame.

The scripts provided scan a slice of the top of the window over a number of frames to detect when inferencing goes from 1 to 0 (similar to detecting a button press). To debounce, an object is only counted if there are N number of 0 values detected after a 1 is.

An example object counting classifier file is provided here (see .eim) from this project: https://studio.edgeimpulse.com/public/165829/latest 

To run live with a webcam use:
`python classify-camera.py <path to classifier file.eim> <camera port>`

To run on an existing video: `python classify-video.py <path to classifier file.eim> <path to .mp4>`

Scripts adapted from https://github.com/edgeimpulse/linux-sdk-python/tree/master/examples/image

Examples:


https://user-images.githubusercontent.com/117161381/206453304-64626574-4540-4baa-a4b1-7077b3656733.mp4



https://user-images.githubusercontent.com/117161381/206453333-5c855718-702c-4520-a778-41caae6f6677.mp4



https://user-images.githubusercontent.com/117161381/206453355-5ad93fe2-c149-45f6-bdf9-c0b073756664.mp4

