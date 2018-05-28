# Natural lighting
# To test with the android doll, run using the following colors:
# python3 main.py 2a7519 6ba257

# To test with the comet voice ball, run using the following colors:
# python3 main.py 7b2803 fb8a33

# Artificial white light
# comet voice ball
# python3 main.py d4680a fec536

import cv2
import numpy as np
import os
import copy


# the below function was taken from
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def main():
    _offset = 50
    windowname = "Live"
    dim_reference_obj_size = 8 # in cm
    logo_height = 34

    reference_scale = 0.3
    reference_logo = cv2.imread(os.path.join("..", "images","itsonus.png"),0)
    reference_logo = cv2.resize(reference_logo, (int(reference_logo.shape[1] * reference_scale),int(reference_logo.shape[0] * reference_scale)))
    reference_logo = cv2.Canny(reference_logo,100,200)
    tH, tW = reference_logo.shape[:2]

    camera = cv2.VideoCapture(0)
    #HOG for human detection
    hog_descriptor = cv2.HOGDescriptor()
    hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    #HAAR cascades for face detection
    face_haarcascade = cv2.CascadeClassifier(os.path.join("..","res","haarcascade_frontalface_default.xml"))
    scales = np.linspace(0.02, 1.0, 20)[::-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    while (True):
        isFrameReadCorrect, frame = camera.read()
        if isFrameReadCorrect:
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.80), int(frame.shape[0] * 0.80)))
            frame_tracked = copy.deepcopy(frame)

            # gray = cv2.cvtColor(frame_tracked, cv2.COLOR_BGR2GRAY)
            found = None
            gray = cv2.cvtColor(frame_tracked, cv2.COLOR_BGR2GRAY)
            for scale in scales:
                resized = cv2.resize(gray, (int(gray.shape[1] * scale),int(gray.shape[0] * scale)))
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                rH = gray.shape[1] / float(resized.shape[1])
                rW = gray.shape[0] / float(resized.shape[0])
                edged = cv2.Canny(resized, 100, 200)
                result = cv2.matchTemplate(edged, reference_logo, cv2.TM_CCOEFF)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, rH, rW)
            logo = None
            logo_dim = None
            if not found is None and found[0] > 5500000.0:
                (_, maxLoc, rH, rW) = found
                (startX, startY) = (int(maxLoc[0] * rW), int(maxLoc[1] * rH))
                (endX, endY) = (int((maxLoc[0] + tW) * rW), int((maxLoc[1] + tH) * rH))
                logo = ((startX+endX)/2, (startY+endY)/2)
                logo_dim = (max(0,startX-_offset), min(endX+_offset,frame_tracked.shape[1]-1), startY, endY)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            if logo_dim != None:
                logo_ratio = logo_height / abs(logo_dim[3]-logo_dim[2])
                (rects, weights) = hog_descriptor.detectMultiScale(frame_tracked, winStride=(4,3), padding=(0, 0), scale=1.06)
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, overlapThresh=0.60)
                if (pick is not None and len(pick) > 0):
                    for (xA, yA, xB, yB) in pick:
                        human_frame = gray[yA:yB, xA:xB]
                        faces = face_haarcascade.detectMultiScale(human_frame, 1.3, 5)
                        if not faces is None and len(faces) == 1:
                            (x, y, face_width, face_height) = faces[0]
                            # cv2.rectangle(frame, (xA+x, yA+y), (xA+x + face_width, yA+y + face_height), (0, 255, 237), 3)
                            yA = max(yA, yA+y)
                    	# estimate human height
                        human_height = (yB-yA) * logo_ratio
                        human_height_ft = int(human_height / 30.48)
                        human_height_inch = int(human_height % 30.48) / 2.54
                        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                        cv2.putText(frame, str(human_height_ft)+" feet "+ "{0:.0f}".format(human_height_inch)+" inches", (xA+10,yA-10), font, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(windowname, frame)
        cv2.moveWindow(windowname, 10, 10)
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27:
            break
    cv2.destroyAllWindows()
    camera.release()

main()