import cv2
import pytesseract
from pytesseract import Output
import imutils
from imutils.video import WebcamVideoStream
 
cap = WebcamVideoStream(src=0).start()
 
while True:
    # Capture frame-by-frame
    frame = cap.read()
    frame = imutils.resize(frame, width=400)
    ocr_result = []

    d = pytesseract.image_to_data(frame, output_type=Output.DICT)
    print(type(d['text']))
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            ocr_result.append(d['text'][i])
            (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # don't show empty text
            if text and text.strip() != "":
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # frame = cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
 
    print(ocr_result)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.stop()
cv2.destroyAllWindows()