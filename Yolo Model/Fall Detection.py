import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone

# Load the model (Ensure yolov10s.pt is in the same directory)
model = YOLO("yolov10s.pt")  

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(f"Mouse Position: {point}")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video file
cap = cv2.VideoCapture('fall5.mp4')

# Load COCO classes
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    count += 1
    if count % 3 != 0: # Skip frames to increase performance
        continue

    frame = cv2.resize(frame, (1020, 600))

    # Run YOLOv10 inference
    results = model(frame)
    
    # Get detection data (x1, y1, x2, y2, confidence, class)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]

        if 'person' in c:
            w = x2 - x1
            h = y2 - y1
            
            # Aspect Ratio Logic: 
            # If width is greater than height, the person is likely lying down (fall).
            if w > h:
                cvzone.putTextRect(frame, "Fall Detected", (x1, y1 - 10), 1, 1, colorR=(0, 0, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cvzone.putTextRect(frame, "Person", (x1, y1 - 10), 1, 1, colorR=(0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()