
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pytesseract
from imutils.object_detection import non_max_suppression
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 125)


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):

            if scoresData[x] < float(0.5):
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


#///////////////////////////////////////////////////////

def OCR():
    
    engine.say("A4 mode press 1 and for medicine mode press 2")
    engine.runAndWait()
    #///////////////////////////////////////////////////////
    ocr_mode = int(input('''Enter the mode of the OCR operation: A4 Papers: 1 Medicine: 2 '''))
    if ocr_mode == 2:
        print("Medicine Mode activated")
        ###################################################################################
        image = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
        print("Image loaded ")
        # ///////////////////////////////////////////////////////
        orig = image.copy()
        (origH, origW) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (int(320), int(320))
        rW = origW / float(newW)
        rH = origH / float(newH)

        # resize the image and grab the new image dimension2s
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("/home/pi/Desktop/OCR_TTS-master/frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)

        (scores, geometry) = net.forward(layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        final_list = []
        text_empty = ''
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            dX = int((endX - startX) * float(0))
            dY = int((endY - startY) * float(0))
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))
            roi = orig[startY:endY, startX:endX]
            ########################################################################

            text = pytesseract.image_to_string(
                roi, config="-l eng --oem 1 --psm 11")
            print("for:" + text)

            text_empty = text_empty +text + " "
        print(text_empty)
        engine.say(text_empty)
        engine.runAndWait()
        engine.stop()
        exit_loop = True
        while exit_loop:
            engine.say("repeat press 2 else press 1")
            engine.runAndWait()
            if "2" == input("choice: "):
                engine.say(text_empty)
                engine.runAndWait()
                engine.stop()
            else:
                exit_loop = False
                engine.stop()


        #############################################################################
        

    if ocr_mode == 1:
        print("A4 Mode")
        # ---------------------------Load Imagge---------------------------#
        img = cv2.imread('1.png', cv2.IMREAD_COLOR)
        # ---------------------------GreyScale Imagge---------------------------#
        # convert to grey to reduce detials
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # /////////////////////////////////////////////////////////////////
        # ---------------------------Filter1 Imagge---------------------------#
        gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
        # /////////////////////////////////////////////////////////////////
        # ---------------------------Thresholding Imagge---------------------------#
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # /////////////////////////////////////////////////////////////////
        # ---------------------------Result---------------------------#
        original = pytesseract.image_to_string(gray, config=' -l eng --oem 1 ')
        print(original)

        engine.say("words detected are "+original)
        engine.runAndWait()
        engine.stop()
        exit_loop = True
        while exit_loop:
            engine.say("repeat press 2 else press 1")
            engine.runAndWait()
            if "2" == input("choice: "):
                engine.say(original)
                engine.runAndWait()
                engine.stop()
            else:
                exit_loop = False
                engine.stop()




# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
#modeldir = os.path("/home/pi/tflite1/sample_tf_lite_model")
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.6)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# print(str(output_details)+"    this output details1")


floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
#     t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    #print(str(output_details)+"    this output details2")

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
        print("this is a floating txt")

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    with open('tvt.txt', 'w+') as file:
        objects=[]
        for i in range(len(scores)):
            
            object_dict={}
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
#                 print("imH= " + str(imH) + " imW=" + str(imW))
#                 print("ymin="+ str(ymin)  + " xmin="+ str(xmin) + " ymax="+ str(ymax) +" xmax=" + str(xmax))
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
#                 x=int(xmin-(xmax/2))
#                 y=int(ymin-(ymax/2))
#                 print("x=" + str(x) + " y=" + str(y))
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = (object_name) # Example:person
                
#                 print(label)
                #object_dict[(label)]=''
                
                
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                if xmin <= imW / 3:
                    W_pos = "left "
                elif xmin <= (imW / 3 * 2):
                    W_pos = "center "
                else:
                    W_pos = "right "

                if ymin <= imH / 3:
                    H_pos = "top "
                elif ymin <= (imH / 3 * 2):
                    H_pos = "mid "
                else:
                    H_pos = "bottom "
                #print("Dict= "+ str(object_dict.keys()))    
                objects.append(H_pos + W_pos + str(label))
        print(objects,file=file)     
               # space = " "
               # file.write(space)
                #file.write(label)
        
    with open('tvt.txt', 'r') as f:
            contents = f.read()
            print("printing the contents of files")
            print(contents)
        
    if cv2.waitKey(25)== ord('w'):
        with open('tvt.txt','r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n", "")

        engine.say(lines)
        engine.runAndWait()
        engine.stop()               
                    
    if cv2.waitKey(25)== ord('o'):  
        OCR()
        
             
        
        
                
#     if cv2.waitKey(25)== ord('w'):
#         with open('tvt.txt','r') as f:
#             lines = f.readlines()
#         for i in range(len(lines)):
#             lines[i] = lines[i].replace("\n", "")
# 
#         engine.say(lines)
#         engine.runAndWait()
#         engine.stop()
            
                
           
    # Draw framerate in corner of frame
#     cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
          

    # Press 'q' to quit
    if cv2.waitKey(25)  == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()





