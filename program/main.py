import cv2
import numpy as np
import os
import copy

def distance(pt1, pt2):
    x1,y1 = pt1
    x2,y2 = pt2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

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
    logo_height = 34

    camera = cv2.VideoCapture(0)

    reference_scale = 0.3
    reference_logo = cv2.imread(os.path.join("..", "images","itsonus.png"),0)
    reference_logo = cv2.resize(reference_logo, (int(reference_logo.shape[1] * reference_scale),int(reference_logo.shape[0] * reference_scale)))
    reference_logo = cv2.Canny(reference_logo,100,200)
    tH, tW = reference_logo.shape[:2]

    #HOG for human detection
    hog_descriptor = cv2.HOGDescriptor()
    hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    #HAAR cascades for face and eye detection
    face_haarcascade = cv2.CascadeClassifier(os.path.join("..","res","haarcascade_frontalface_default.xml"))
    eye_haarcascade = cv2.CascadeClassifier(os.path.join("..","res","haarcascade_eye.xml"))

    scales = np.linspace(0.02, 1.0, 20)[::-1]
    while (True):
        isFrameReadCorrect, frame = camera.read()
        if isFrameReadCorrect:
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.80), int(frame.shape[0] * 0.80)))
            frame_tracked = copy.deepcopy(frame)
            # detect the logo in the frame and restrict the window only to that person wearing the t-shirt
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
                frame_tracked[0:frame_tracked.shape[0],0:logo_dim[0]] = [0,0,0]
                frame_tracked[0:frame_tracked.shape[0],logo_dim[1]:] = [0,0,0]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            if not (logo is None):

                # detect eyes and face of the person in the window
                faces = face_haarcascade.detectMultiScale(gray, 1.3, 5)
                closest_face = None
                if not faces is None and len(faces) > 0:
                    min_dist = 1000000000000
                    for (x, y, face_width, face_height) in faces:
                        face = (x+(face_width)/2, y+(face_height)/2)
                        dist = distance(logo, face)
                        if (dist < min_dist and logo[1] > face[1] and (face[0] >= logo_dim[0] and face[0] <= logo_dim[1])):
                            min_dist = dist
                            closest_face = (x,y,face_width,face_height)
                    if closest_face is not None:
                        (x, y, face_width, face_height) = closest_face
                        cv2.rectangle(frame, (x, y), (x + face_width, y + face_height), (0, 255, 237), 3)
                        frame_tracked[0:max(0,y-_offset),0:frame_tracked.shape[1]] = [0,0,0]
                        roi_gray = gray[y:y + face_height, x:x + face_width]
                        roi_color = frame[y:y + face_height, x:x + face_width]
                        eyes = eye_haarcascade.detectMultiScale(roi_gray)
                        for (eye_x, eye_y, eye_width, eye_height) in eyes:
                                cv2.circle(roi_color, (int(eye_x+(eye_width/2)), int(eye_y+(eye_height/2))), int(max(eye_width,eye_height)/2), (255, 229, 0), 3)


                # detect the people in the window precisely
                (rects, weights) = hog_descriptor.detectMultiScale(frame_tracked, winStride=(4,3), padding=(0, 0), scale=1.06)
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, overlapThresh=0.80)
                
                if (pick is not None and len(pick) > 0):
                    (max_xA, max_yA, max_xB, max_yB) = pick[0]
                    max_area = abs(max_xB-max_xA)*abs(max_yB-max_yA)
                    for (xA, yA, xB, yB) in pick:
                        area = abs(xB-xA)*abs(yB-yA)
                        if (area > max_area):
                            max_area = area
                            (max_xA, max_yA, max_xB, max_yB) = (xA-50,yA,xB,yB)
                    if closest_face is not None:
                        max_yA = max(max_yA, closest_face[1])

                    # estimate human height
                    human_height = (max_yB-max_yA) * logo_height / abs(logo_dim[3]-logo_dim[2])
                    human_height_ft = int(human_height / 30.48)
                    human_height_inch = int(human_height % 30.48) / 2.54
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.rectangle(frame, (max_xA, max_yA), (max_xB, max_yB), (0, 255, 0), 2)
                    cv2.putText(frame, str(human_height_ft)+" feet "+ "{0:.0f}".format(human_height_inch)+" inches", (max_xA,max_yA-10), font, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(windowname, frame)
        cv2.moveWindow(windowname, 10, 10)
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27:
            break
    cv2.destroyAllWindows()
    camera.release()

main()