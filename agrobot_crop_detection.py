import cv2
from ultralytics import YOLO
import random

def generate_random_color():
  b = random.randint(0, 255)
  g = random.randint(0, 255)
  r = random.randint(0, 255)
  return (b, g, r)

model = YOLO('agronomic_model_20_march_2024.pt', 'cuda:0')

# set model parameters
model.overrides['conf'] = 0.75  # NMS confidence threshold
model.overrides['iou'] = 0.75  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

min_area = 100

cap = cv2.VideoCapture(0)

# frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(1152))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(648))

font = cv2.FONT_HERSHEY_SIMPLEX

class_colors = {}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Can't receive frame from video stream")
        break

    # Skip processing empty frames (optional)
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Skipping empty frame")
        continue

    # Perform YOLOv8 object detection
    results = model(frame)

    # Loop through each image in results
    for result in results:
        # Get bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        class_ids = result.boxes.cls

        # Loop through detections
        for i, (box, score, class_id) in enumerate(zip(boxes, conf, class_ids)):
            # Get object information
            x_min, y_min, x_max, y_max = box
            confidence = score

            # Convert class_id to integer
            class_id = int(class_id)

            # Filter detections based on confidence threshold
            if confidence > 0.6:
                # Calculate bounding box area
                area = (x_max - x_min) * (y_max - y_min)

                # Filter detections based on minimum area (optional)
                if area > min_area:
                    # Check if class_id exists in model.names dictionary
                    if class_id not in model.names:
                        print(f"Warning: Class ID {class_id} not found in model.names dictionary.")
                        continue

                    # Get class name corresponding to class ID
                    class_name = model.names[class_id]

                    # Generate random color if not already in dictionary
                    if class_name not in class_colors:
                        class_colors[class_name] = generate_random_color()

                    # Get color from dictionary
                    color = class_colors[class_name]

                    # Draw colored bounding box and label
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    label = f"{class_name}: {confidence:.2f}"  # Display name and confidence
                    cv2.putText(frame, label, (x_min + 5, y_min - 5), font, 0.7, color)  # Text color matches box

    # Display the resulting frame
    cv2.imshow('AgRobot: Crops Detection for Farming Environment by Ansh Ranjan', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()