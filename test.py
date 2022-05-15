import numpy as np
import cv2
import argparse
from imutils.object_detection import non_max_suppression
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import time

# import datetime
# from threading import Thread

# class FPS:
#     def __init__(self):
#         # store the start time, end time, and total number of frames
#         # that were examined between the start and end intervals
#         self._start = None
#         self._end = None
#         self._numFrames = 0
#     def start(self):
#         # start the timer
#         self._start = datetime.datetime.now()
#         return self
#     def stop(self):
#         # stop the timer
#         self._end = datetime.datetime.now()
#     def update(self):
#         # increment the total number of frames examined during the
#         # start and end intervals
#         self._numFrames += 1
#     def elapsed(self):
#         # return the total number of seconds between the start and
#         # end interval
#         return (self._end - self._start).total_seconds()
#     def fps(self):
#         # compute the (approximate) frames per second
#         return self._numFrames / self.elapsed()

# class WebcamVideoStream:
#     def __init__(self, src=0):
#         # initialize the video camera stream and read the first frame
#         # from the stream
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         # initialize the variable used to indicate if the thread should
#         # be stopped
#         self.stopped = False
        
#     def start(self):
#         # start the thread to read frames from the video stream
#         Thread(target=self.update, args=()).start()
#         return self
#     def update(self):
#         # keep looping infinitely until the thread is stopped
#         while True:
#             # if the thread indicator variable is set, stop the thread
#             if self.stopped:
#                 return
#             # otherwise, read the next frame from the stream
#             (self.grabbed, self.frame) = self.stream.read()
#     def read(self):
#         # return the frame most recently read
#         return self.frame
#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    help="path to input image")
ap.add_argument("-east", "--east", type=str, default='frozen_east_text_detection.pb',
    help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
    help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
    help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
    help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
    help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

stream = WebcamVideoStream(src=0).start()
# fps = FPS().start()
# if not stream.isOpened():
#     print("Cannot open camera")
#     exit()

ocr_start = time.time()
while True:
    # Capture frame-by-frame
    frame = stream.read()
    frame = imutils.resize(frame, width=400)
    # frame = cv2.resize(frame, (239, 179))
    # if frame is read correctly ret is True
    # if not ret:
    #     print("Can't receive frame (stream end?). Exiting ...")
    #     break
    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    orig = frame.copy()
    (origH, origW) = frame.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    frame = cv2.resize(frame, (newW, newH))
    (H, W) = frame.shape[:2]
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    if time.time() - ocr_start > 1:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        if(not isinstance(boxes, list)):
            box = np.array(
            [np.amin(boxes, axis=0)[0], np.amin(boxes, axis=0)[1], 
            np.amax(boxes, axis=0)[2], np.amax(boxes, axis=0)[3]]
            )
            # loop over the bounding boxes
            startX, startY, endX, endY = box
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])
            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))
# When everything done, release the capture
stream.stop()
cv2.destroyAllWindows()