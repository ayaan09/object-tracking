import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
output_layers = net.getUnconnectedOutLayersNames()

# Initialize video capture
input_video = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/Front Out 1.MOV"
output_video = "output_video.MOV"
cap = cv2.VideoCapture(input_video)
codec = cv2.VideoWriter_fourcc(*"XVID")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter(output_video, codec, fps, (width, height))

# Initialize variables for person tracking
person_count = 0
person_dict = {}

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extract bounding box information
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9 and class_id == 0:  # Class ID 0 represents people
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and tags
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = "Person " + str(person_count + 1)
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 5), font, 1, color, 2)
            
            # Store the person's bounding box coordinates for tracking
            person_dict[label] = (x, y, x + w, y + h)
            
            person_count += 1

    # Update existing person tags based on tracking
    for person, (x1, y1, x2, y2) in person_dict.items():
        if person in person_dict and person_count > 0:
            try:
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[person_count - 1], 2)
                cv2.putText(frame, person, (x1, y1 - 5), font, 1, colors[person_count - 1], 2)
            except:
                pass

    # Write frame to output video
    output.write(frame)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()