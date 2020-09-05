import cv2 as cv
import numpy as np
import utils 
import time
from imutils.object_detection import non_max_suppression
import pytesseract


model_path = "Vision/dnn/frozen_east_text_detection.pb"

text_detector = cv.dnn.readNet(model_path)
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]


cap = utils.open_camera()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    (H,W) = frame.shape[:2]
    small_frame = cv.resize(frame,(320,320))
    (smallH,smallW) = small_frame.shape[:2]

    rW = W / float(smallW)
    rH = H / float(smallH)
    #preform text detection here
    blob = cv.dnn.blobFromImage(small_frame, 1.0, (smallW, smallH),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    text_detector.setInput(blob)
    (scores, geometry) = text_detector.forward(layerNames)
    end = time.time()

    #print("scores: ",np.shape(scores))

    
    (rows,cols) = scores.shape[2:4]
    rects = []
    probs = []
    # loop over the number of rows
    for y in range(0, rows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, cols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            #print(scoresData[x])
            rects.append((startX, startY, endX, endY))
            probs.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=probs)

    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        region = frame[max(min(startY, H),0):max(min(endY, H), 0),max(min(startX, W), 0):max(min(endX, W), 0)]
        configuration = ("-l eng --oem 1 --psm 8")
        text = pytesseract.image_to_string(region,config=configuration)
        cv.putText(frame, text, (startX, startY - 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)
        cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()