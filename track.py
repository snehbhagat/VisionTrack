import numpy as np
import argparse
import cv2

# Initialize the current frame of the video, along with the list of ROI points and whether or not this is input mode
frame = None
roiPts = []  # region of interest points
inputMode = False
tracker = None

def selectROI(event, x, y, flags, param):
    global frame, roiPts, inputMode
    
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

def main():
    global frame, roiPts, inputMode, tracker
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())
    
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)
    
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break
        
        if tracker is not None:
            # Update the tracker and get the new position
            success, box = tracker.update(frame)
            if success:
                # Draw the bounding box
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("i") and len(roiPts) < 4:
            inputMode = True
            orig = frame.copy()
            while len(roiPts) < 4:
                cv2.imshow("frame", frame)
                cv2.waitKey(0)
            
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis=1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            
            # Initialize the CSRT tracker with the first frame and the bounding box
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]))
        
        elif key == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()