from ultralytics import YOLO
import cv2
from gtts import gTTS
from playsound import playsound
import os

def speech(text):
    try:
        print(text)
        language = "en"
        output = gTTS(text=text, lang=language, slow=False)

        # Ensure the directory exists
        if not os.path.exists("./sounds"):
            os.makedirs("./sounds")

        output.save("./sounds/output.mp3")
        playsound("./sounds/output.mp3")
    except Exception as e:
        print(f"Error with TTS or audio playback: {e}")

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')  # Adjust to other variants like yolov8s.pt if needed

# Open video file instead of webcam feed
video_path = 'path_to_your_video.mp4'  # Replace with your video file path
video = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not video.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

labels = []

# Define confidence threshold
confidence_threshold = 0.5

while True:
    ret, frame = video.read()
    if not ret:
        break  # Video has ended

    # Perform object detection
    results = model(frame)
    
    # Extract labels from the detection results with high confidence
    detected_boxes = results[0].boxes
    detected_labels = [model.names[int(box.cls)] for box in detected_boxes if box.conf > confidence_threshold]

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    for item in detected_labels:
        if item not in labels:
            labels.append(item)
            speech(f"New object detected: {item}")

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# Generate final speech summary for detected objects
if labels:
    speech(f"I found {', '.join(labels[:-1])} and {labels[-1]}.")
else:
    speech("No objects detected.")
