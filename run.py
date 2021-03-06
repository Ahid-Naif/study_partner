from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import time
import keyboard
import queue # تنظيم قراءة الصوت لعدم ضياع البيانات
import sounddevice as sd # قراءة الصوت من الميكروفون
import vosk # تحويل الصوت إلى نص
import sys
import json
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import re
from textblob import TextBlob
from urllib.error import HTTPError
import tkinter as tk

def printResultsMsgBox(text1, text2):
    root = tk.Tk()
    frame1 = tk.Frame(root, bg='gold')
    frame1.pack(fill=tk.X)
    frame1.columnconfigure(0, weight=1)
    frame2 = tk.Frame(root)
    frame2.pack()
    frame3 = tk.Frame(root, bg='gold')
    frame3.pack(fill=tk.X)
    frame3.columnconfigure(0, weight=1)
    frame4 = tk.Frame(root)
    frame4.pack()
    frame1label1 = tk.Label(frame1, bg='gold', text='OCR Result')
    frame1label1.grid(row=0, column=0)
    frame2label = tk.Label(frame2, text=text1)
    frame2label.pack()
    frame3label1 = tk.Label(frame3, bg='gold', text='Voice Result')
    frame3label1.grid(row=1, column=0)
    frame4label = tk.Label(frame4, text=text2)
    frame4label.pack()
    root.mainloop()
    root.quit()

def welcomeScreen():
    global status
    print('Press (y) to start')
    cap = cv2.VideoCapture('screens2/welcome.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read() 
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
        if ret:
            cv2.imshow("window", frame)
        else:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           continue

        if cv2.waitKey(1) & 0xFF == ord('y'):
            cap.release()
            cv2.destroyAllWindows()
            status = 'camera'
            main()
        time.sleep(0.03333) # 30 fps

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

def check1(text):
    global status
    # display the text OCR'd by Tesseract
    print("TEXT")
    print("========")
    print("{}\n".format(text))

    print('Press (y) for yes, or (n) for no')
    while True:
        if keyboard.read_key() == "y":
            ocr_file = open("ocr.txt", "w")
            ocr_file.write(text)
            ocr_file.close()
            main()
        elif keyboard.read_key() == "n":
            status = 'camera'
            main()

def ocrProgram():
    global status
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
    if not vs.isOpened():
        print("Cannot open camera")
        exit()
    isOCR = ''
    while True:
        # Capture frame-by-frame
        ret, frame = vs.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frames")
            break
        
        # Our operations on the frame come here
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

        output = orig.copy()
        if(not isinstance(boxes, list)):
            if isOCR == False:
                cv2.destroyWindow('Video')

            cv2.rectangle(output, (startX, startY), (endX, endY),
            	(0, 0, 255), 2)
            cv2.namedWindow("OCR", cv2.WND_PROP_FULLSCREEN) 
            cv2.setWindowProperty("OCR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('OCR', output)
            isOCR = True

            key = cv2.waitKey(1)
            if key == ord("y"):
                status = 'voice'
                break
            elif key == ord("n"):
                status = 'welcome'
                vs.release()
                cv2.destroyAllWindows()
                main()
        else:
            if isOCR == True:
                cv2.destroyWindow('OCR')

            # show the output image
            cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN) 
            cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Video", output)
            isOCR = False

            key = cv2.waitKey(1)
            if key == ord("n"):
                status = 'welcome'
                vs.release()
                cv2.destroyAllWindows()
                main()

    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 1, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case
    config = ("-l eng --oem 1 --psm 3")
    text = pytesseract.image_to_string(roi, config=config)
    tb = TextBlob(text)
    text = str(tb.correct())
    
    vs.release()
    cv2.destroyAllWindows()
    check1(text)

# تعريف ال functions
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def voiceProgram():
    results_voice = []
    img = cv2.imread('screens2/mic.jpg')
    while True:
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", img)
        if cv2.waitKey(1) & 0xFF == 32:
            break
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture('screens2/listening.mp4')
    soundStream.start()
    start_time = time.time()
    while(cap.isOpened()):
        if time.time() - start_time > 0.03333:  # 30 fps
            ret, frame = cap.read() 
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

            if ret:
                cv2.imshow("window", frame)
            else:
               cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
               continue
            start_time = time.time()
        
        data = q.get()
        if rec.AcceptWaveform(data):
            sentence = rec.Result() # تم تحويل الصوت إلى نص
            sentence = json.loads(sentence)
            results_voice.append(sentence.get("text", ""))
        else:
            partial = rec.PartialResult() # تم تحويل الصوت إلى نص
        
        if cv2.waitKey(1) & 0xFF == 32:
            start_time2= time.time()
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    sentence = rec.Result() # تم تحويل الصوت إلى نص
                    sentence = json.loads(sentence)
                    results_voice.append(sentence.get("text", ""))
                else:
                    partial = rec.PartialResult() # تم تحويل الصوت إلى نص
            
                if time.time() - start_time2 > 2:
                    break
            cap.release()
            cv2.destroyAllWindows()
            results_voice = " ".join(results_voice)
            soundStream.stop()
            check2(results_voice)

def check2(results_voice):
    global status
    # display the text OCR'd by Tesseract
    print("TEXT")
    print("========")
    print("{}\n".format(results_voice))

    print('Press (y) for yes, or (n) for no')
    while True:
        if keyboard.read_key() == "y":
            speech_file = open("speech.txt", "w")
            speech_file.write(results_voice)
            speech_file.close()
            status = 'result'
            main()

        elif keyboard.read_key() == "n":
            voiceProgram()

def similarityProgram():
    global status
    global waiting
    global pressed_time
    global pressed
    global long_pressed
    #open text file in read mode
    ocr = open("ocr.txt", "r")
    ocr_data = ocr.read()
    ocr.close()
    speech = open("speech.txt", "r")
    speech_data = speech.read()
    speech.close()
    
    ocr_data = re.sub(r'[^\w]', ' ', ocr_data).rstrip()
    speech_data = re.sub(r'[^\w]', ' ', speech_data).rstrip()
    print("Image Result")
    print("========")
    print("{}\n".format(ocr_data))
    print("Voice Result")
    print("========")
    print("{}\n".format(speech_data))

    cap = cv2.VideoCapture('screens2/processing.mp4')
    start_time = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imshow("window", frame)
            cv2.waitKey(1)
            time.sleep(0.03333)
        else:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           continue

        if time.time() - start_time > 3:
            cap.release()
            cv2.destroyAllWindows()
            break
        
    my_result_out = session.run(
    my_result, feed_dict={text_input: [ocr_data.lower(), speech_data.lower()]})
    corr = np.inner(my_result_out, my_result_out)
    result = float("{:.2f}".format(corr[0][1]))*100


    if result > 70.0:
        cap2 = cv2.VideoCapture('screens2/happy.mp4')
    elif result >= 40.0 and result <= 70.0:
        cap2 = cv2.VideoCapture('screens2/neutral.mp4')
    elif result < 40.0:
        cap2 = cv2.VideoCapture('screens2/sad.mp4')

    while(cap2.isOpened()):
        ret, frame = cap2.read() 
        if ret:
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", frame)
        else:
           cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
           continue
        time.sleep(0.09999) # 30 fps
        if cv2.waitKey(1) & 0xFF == ord('y'):
            break

    cap.release()
    cv2.destroyAllWindows()

    printResultsMsgBox(ocr_data, speech_data)

    status = 'welcome'
    main()

def main():
    while True:
        if status == 'welcome':
            welcomeScreen()
        elif status == 'camera':
            ocrProgram()
        elif status == 'voice':
            voiceProgram()
        elif status == 'result':
            similarityProgram()

if __name__ == '__main__':
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
    
    ## Vosk
    # تعريف مكتبة queue
    q = queue.Queue()
    samplerate = 48000
    model = vosk.Model("model")
    rec = vosk.KaldiRecognizer(model, samplerate)
    rec.SetWords(True)
    soundStream = sd.RawInputStream(samplerate=samplerate, blocksize = 8000, dtype='int16', channels=1, callback=callback)
    ## Vosk - End
    
    # Similarity
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    module_url = ROOT_DIR+"\module"
    
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.load(module_url)
        my_result = embed(text_input)
        # model_file = open("model.txt", "w")
        # model_file.write(my_result)
        # model_file.close()
        init_op = tf.group(
            [tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()
    
    # Create session and initialize.
    session = tf.Session(graph=g)
    session.run(init_op)
    ## Similarity - End

    status = 'welcome' # welcome, camera, voice, result
    main()