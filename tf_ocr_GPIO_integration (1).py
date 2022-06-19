# ## Import Standard Python
import os
import sys
import time
# ## 
import numpy as np
from threading import Thread
import tensorflow as tf
from PIL import Image
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression
import argparse
import RPi.GPIO as GPIO
from time import sleep


# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util
# ## Import TTS module
import pyttsx3
engine = pyttsx3.init()

engine.setProperty('rate', 125)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(13,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(15,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)





class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
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
## End of class





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
# Initialize video stream
videostream = VideoStream(resolution=(1280,720),framerate=30).start()
time.sleep(1)
imW = 1280
imH = 720

def OCR():
    
    cv2.destroyAllWindows()
    engine.say("A4 mode press 13 and for medicine mode press 11")
    engine.runAndWait()
    engine.stop()
    #///////////////////////////////////////////////////////
    #ocr_mode = int(input('''Enter the mode of the OCR operation: A4 Papers: 1 Medicine: 2 '''))
    ocr_run = True
    while ocr_run:
        sleep(.25)
        if GPIO.input(11)==GPIO.HIGH:
            frame=videostream.read()
            cv2.imshow('frame',frame)
            cv2.imwrite("1.jpg",frame)
            
            print("Medicine Mode activated")
            

            ###################################################################################
            engine.say("Medicine Mode activated")
            image = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
            engine.say("Image loaded")
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
            sleep(1)
            engine.say("repeat press the same button ")
            engine.runAndWait()
            engine.stop()

            while exit_loop:
               
                sleep(.25)
                if GPIO.input(11)==GPIO.HIGH:

                    engine.say(text_empty)
                    engine.runAndWait()
                    engine.stop()
                elif GPIO.input(13)==GPIO.HIGH:
                     ocr_run=False
                     
                    
                     exit_loop = False
                     engine.stop()
                    
                else:
                    pass    


            #############################################################################
        elif GPIO.input(13) == GPIO.HIGH:
            frame=videostream.read()
            cv2.imwrite("1.png",frame)
            engine.say("A4 Mode activated")
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
            
            sleep(1)
            engine.say("repeat press the same button")
            engine.runAndWait()
            engine.stop()
            
            while exit_loop:
                sleep(.25)
                if GPIO.input(13)==GPIO.HIGH:

                    engine.say(original)
                    engine.runAndWait()
                    engine.stop()
                elif GPIO.input(11)==GPIO.HIGH:
                    ocr_run = False
                    exit_loop = False
                    engine.stop()
                    
                else:
                    pass
                     
        elif (GPIO.input(15)==GPIO.HIGH):
            break
        
        else:
            pass
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph_ssdlite.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_path = "/home/pi/tensorflow1/Object_detection_project/"+MODEL_NAME+"/graph.pbtxt"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
                    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            image_np = videostream.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25)&0xFF==ord('w'):
                pass
            # End of visualization  #
            lst = np.squeeze(boxes)
            _boxes = lst[[x for x, i in enumerate(lst) if i.any()]] # remove np_array[0 0 0 0] from np.sqqueeze(boxes)
            objects = []
            i,j = 1,0

            for index, value in enumerate(classes[0]):
                i += 1
                if scores[0, index] > 0.5:
                    # # prcess to get dimensions:
                    ymin = int(max(1,(_boxes[j][0] * imH)))
                    xmin = int(max(1,(_boxes[j][1] * imW)))
                    ymax = int(min(imH,(_boxes[j][2] * imH)))
                    xmax = int(min(imW,(_boxes[j][3] * imW)))
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
                    ##

                    objects.append(H_pos + W_pos + str((category_index.get(value)).get('name')))
                    j += 1
                print(objects)
            sleep(.25)
            if GPIO.input(11)==GPIO.HIGH:

                
                engine.say(objects)
                engine.runAndWait()
                engine.stop()
            sleep(.25)
            if GPIO.input(13)==GPIO.HIGH:

                OCR()      

            




# Log File : to document new SW updates to Object detcection feature "Graduation project"
# #v002: [Adding TTS module, remove dictionary and save object direct to list, process to visualize the position of objects]
# #Note v002: position is not accurate according to the np.squeeze[] arrays output in sequence
# #v002: TTS engine read direct from object[], removing read/write object into/from tvt.txt  
