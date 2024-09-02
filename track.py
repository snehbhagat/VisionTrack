# import the necessary packages
import numpy as np
import argparse
import cv2
# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts = [] # region of interest points
inputMode = False
def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode
    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
    args = vars(ap.parse_args())
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode
    # if the video path was not supplied, grab the reference to the
    # camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
        # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
        # setup the mouse callback
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", selectROI)
        # initialize the termination criteria for cam shift, indicating
        # a maximum of ten iterations or movement by a least one pixel
        # along with the bounding box of the ROI
        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        roiBox = None