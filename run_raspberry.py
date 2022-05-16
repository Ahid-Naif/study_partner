# from glob import glob
from glob import glob
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import time
# import keyboard
import queue # تنظيم قراءة الصوت لعدم ضياع البيانات
import sounddevice as sd # قراءة الصوت من الميكروفون
import vosk # تحويل الصوت إلى نص
import sys
import json
import re
from textblob import TextBlob
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import requests
from urllib.error import HTTPError
from imutils.video import WebcamVideoStream
from pytesseract import Output
import imutils

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BCM) # Use physical pin numbering
GPIO.setup(5, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 5 to be an input pin and set initial value to be pulled low (off)

waiting = False
pressed = False
long_pressed = False
pressed_time = time.time()
start_time = time.time()

def welcomeScreen():
    global status
    global waiting
    global pressed_time
    global pressed
    global long_pressed
    cap = cv2.VideoCapture('screens2/welcome.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read() 
        # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
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
            # break
        time.sleep(0.03333) # 30 fps

        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
        else:
            if pressed:
                pressed = False
                long_pressed = False
                cap.release()
                cv2.destroyAllWindows()
                status = 'camera'
                main()
    
            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True
    # while True:
        # if keyboard.read_key() == "y":
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     status = 'camera'
        #     main()

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
    global waiting
    global pressed_time
    global pressed
    global long_pressed
    # display the text OCR'd by Tesseract
    print("TEXT")
    print("========")
    print("{}\n".format(text))

    print('Press (y) for yes, or (n) for no')
    while True:
        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
                    time.sleep(1)
        else:
            if pressed:
                if long_pressed:
                    status = 'camera'
                    main()
                else:
                    ocr_file = open("ocr.txt", "w")
                    ocr_file.write(str(text))
                    ocr_file.close()
                    pressed = False
                    long_pressed = False
                    main()

            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True
        # if keyboard.read_key() == "y":
        #     ocr_file = open("ocr.txt", "w")
        #     ocr_file.write(text)
        #     ocr_file.close()
        #     main()
        # elif keyboard.read_key() == "n":
        #     status = 'camera'
        #     main()

def ocrProgram():
    global status
    global waiting
    global pressed_time
    global pressed
    global long_pressed
    global vs
    global n_boxes
    global ocr_result = []
    print("[INFO] starting video stream...")
    vs.start()
    start_time = time.time() -10   
    while True:
        # Capture frame-by-frame
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        if time.time() - start_time > 5:
            ocr_result = []
            d = pytesseract.image_to_data(frame, output_type=Output.DICT)
            n_boxes = len(d['text'])
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    ocr_result.append(d['text'][i])
                    (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    # don't show empty text
                    if text and text.strip() != "":
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ocr_result = " ".join(ocr_result)
            tb = TextBlob(ocr_result)
            text = tb.correct()
            start_time = time.time()
        else:
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    ocr_result.append(d['text'][i])
                    (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    # don't show empty text
                    if text and text.strip() != "":
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ocr_result = " ".join(ocr_result)
            tb = TextBlob(ocr_result)
            text = tb.correct()
            start_time = time.time() 

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
        else:
            if pressed:
                pressed = False
                long_pressed = False
                break

            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True

    vs.stop()
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
    global waiting
    global pressed_time
    global pressed
    global long_pressed
    img = cv2.imread('screens2/mic.jpg')
    while True:
        cv2.imshow("window", img)
        cv2.waitKey(1)

        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
        else:
            if pressed:
                pressed = False
                long_pressed = False
                break

            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True
        # if cv2.waitKey(1) & 0xFF == 32:
        #     break
    cv2.destroyAllWindows()
    print("break")

    cap = cv2.VideoCapture('screens2/listening.mp4')
    soundStream.start()
    start_time = time.time()
    while(cap.isOpened()):
        # print("loop")
        if time.time() - start_time > 0.03333:  # 30 fps
            ret, frame = cap.read() 
            # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

            if ret:
                cv2.imshow("window", frame)
                cv2.waitKey(1)
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
        
        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
        else:
            if pressed:
                pressed = False
                long_pressed = False
                time.sleep(1)
                cap.release()
                cv2.destroyAllWindows()
                results_voice = " ".join(results_voice)
                soundStream.stop()
                check2(results_voice)

            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True
        # if cv2.waitKey(1) & 0xFF == 32:
        #     start_time2= time.time()
        #     while True:
        #         data = q.get()
        #         if rec.AcceptWaveform(data):
        #             sentence = rec.Result() # تم تحويل الصوت إلى نص
        #             sentence = json.loads(sentence)
        #             results_voice.append(sentence.get("text", ""))
        #         else:
        #             partial = rec.PartialResult() # تم تحويل الصوت إلى نص
            
        #         if time.time() - start_time2 > 2:
        #             break
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     results_voice = " ".join(results_voice)
        #     soundStream.stop()
        #     check2(results_voice)

def check2(results_voice):
    global status
    global waiting
    global pressed_time
    global pressed
    global long_pressed
    # display the text OCR'd by Tesseract
    print("TEXT")
    print("========")
    print("{}\n".format(results_voice))

    print('Press (y) for yes, or (n) for no')
    while True:
        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
        else:
            if pressed:
                if long_pressed:
                    voiceProgram()
                else:
                    speech_file = open("speech.txt", "w")
                    speech_file.write(results_voice)
                    speech_file.close()
                    status = 'result'
                    main()
                pressed = False
                long_pressed = False

            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True
        # if keyboard.read_key() == "y":
        #     # speech_file = open("speech.txt", "w")
        #     # speech_file.write(results_voice)
        #     # speech_file.close()
        #     # status = 'result'
        #     # main()

        # elif keyboard.read_key() == "n":
        #     voiceProgram()

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

    myobj = {'a': ocr_data, 'b': speech_data}
    myobj =  json.dumps(myobj)

    try:
        x = requests.post('https://hammerhead-app-lnodz.ondigitalocean.app/similar', data = myobj)
    except HTTPError as e:
        pass
    time.sleep(5)
   
    cap = cv2.VideoCapture('screens2/processing.mp4')
    start_time = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imshow("window", frame)
        else:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           continue
        
        if time.time() - start_time > 5:
            try:
                x = requests.get('https://hammerhead-app-lnodz.ondigitalocean.app/getResult')
                result = x.text
                if result != "":
                    result = float(x.text)
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            except HTTPError as e:
                print(e)    
            start_time = time.time()
           
        time.sleep(0.03333) # 30 fps
        cv2.waitKey(1)

    if result > 70.0:
        cap2 = cv2.VideoCapture('screens2/happy.mp4')
    elif result >= 40.0 and result <= 70.0:
        cap2 = cv2.VideoCapture('screens2/neutral.mp4')
    elif result < 40.0:
        cap2 = cv2.VideoCapture('screens2/sad.mp4')

    while(cap2.isOpened()):
        ret, frame = cap2.read() 
        if ret:
            cv2.imshow("window", frame)
        else:
           cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
           continue
        time.sleep(0.09999) # 30 fps
        cv2.waitKey(1)
        if waiting:
            if(time.time() - pressed_time)*1000 >= 20:
                if GPIO.input(5) == GPIO.HIGH:
                    pressed = True
            if time.time() - pressed_time > 1:
                waiting = False
                if GPIO.input(5) == GPIO.HIGH:
                    long_pressed = True
        else:
            if pressed:
                break

            elif GPIO.input(5) == GPIO.HIGH:
                pressed_time = time.time()
                waiting = True

    status = 'welcome'
    main()

def main():
    while True:
        ocrProgram()
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

    vs = WebcamVideoStream(src=0)
    
    ## Similarity
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # module_url = ROOT_DIR+"/module"
    
    # g = tf.Graph()
    # with g.as_default():
    #     text_input = tf.placeholder(dtype=tf.string, shape=[None])
    #     embed = hub.load(module_url)
    #     my_result = embed(text_input)
    #     init_op = tf.group(
    #         [tf.global_variables_initializer(), tf.tables_initializer()])
    # g.finalize()
    
    # # Create session and initialize.
    # session = tf.Session(graph=g)
    # session.run(init_op)
    ## Similarity - End

    status = 'welcome' # welcome, camera, voice, result
    main()