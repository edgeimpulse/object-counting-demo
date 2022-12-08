#!/usr/bin/env python

#import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import itertools
from collections import deque
from statistics import mode
import os
import time
import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a video preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False


def help():
    print('python classify-video.py <path_to_model.eim> <path_to_video.mp4>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) != 2:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            count = 0
            vidcap = cv2.VideoCapture(args[1])
            sec = 0
            start_time = time.time()

            def getFrame(sec):
                vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
                hasFrames,image = vidcap.read()
                if hasFrames:
                    return image
                else:
                    print('Failed to load frame', args[1])
                    exit(1)


            img = getFrame(sec)
            # Change x and y windows to detect smaller items (limited by FOMO windowing)
            x_windows = 10
            y_windows = 10
            # forward_buffer-1 == how many frames have no object after one is detected to count
            forward_buffer = 2
            detections = np.zeros((x_windows,y_windows))
            frame_queue = deque(maxlen=forward_buffer)
            frame_queue.append(detections[0,:])
        
            
            
            while img.size != 0:
                # imread returns images in BGR format, so we need to convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                features, cropped = runner.get_features_from_image(img)
                detections = np.zeros((x_windows,y_windows))
                

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                res = runner.classify(features)

                
                if "bounding_boxes" in res["result"].keys():
                    #print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    #Convert found items into points on grid
                    for bb in res["result"]["bounding_boxes"]:
                        img = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                        # map x and y values to the windowed array
                        x=bb['x']+(bb['width']/2)
                        x_n=round(np.interp(x, [0,cropped.shape[0]],[0,x_windows-1]))
                        y=bb['y']+(bb['height']/2)
                        y_n=round(np.interp(y, [0,cropped.shape[1]],[0,y_windows-1]))
                        detections[y_n,x_n]=1

                # Detect if items have left the top row and debounce
                top_row = detections[0,:]
                frame_queue.append(top_row)
                for column, value in enumerate(frame_queue[0]):
                    debounced = all(ele[column] == 0 for ele in itertools.islice(frame_queue,1,forward_buffer))
                    if value == 1 and debounced:
                        count +=1



                if (show_camera):
                    im2 = cv2.resize(img, dsize=(800,800))
                    cv2.putText(im2, f'{count} items passed', (15,750), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    cv2.imshow('edgeimpulse', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),dsize=(500,500))
                    print(top_row)
                    if cv2.waitKey(1) == ord('q'):
                        break

                sec = time.time() - start_time
                sec = round(sec, 2)
                # print("Getting frame at: %.2f sec" % sec)
                img = getFrame(sec)
        finally:
            if (runner):
                print(f'{count} Items Left Conveyorbelt')
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])