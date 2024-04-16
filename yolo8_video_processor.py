import os
from PIL import Image
from ultralytics import YOLO
import cv2 

model = YOLO("yolov8x")


# Path to the folder containing the images
image_folder_path = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/videos/data/"


result = model.track('C:/Users/Hanzalah Choudhury/Desktop/boundingbox/videos/station1_in.mp4',classes=[0], device = 'cuda:0' , conf = 0.91, iou=0.5, save=False, tracker='bytetrack.yaml')

i=-1
for r in result:
    id_str = r.boxes.id
    bbox_ = r.boxes.xywhn
    conf_ = r.boxes.conf
    i+=1
    if(id_str!=None):
        id_str = id_str.tolist()
        bbox_ = bbox_.tolist()   
        conf = conf_.tolist()
            # Construct the image file path
        image_file_path = os.path.join(image_folder_path, f"frame{i}.jpg")  # Adjust the file extension if necessary
        print(image_file_path)
            # Check if the image file exists
        if os.path.exists(image_file_path):
                # Open the image
            image = cv2.imread(image_file_path)
            (height, width) = image.shape[:2]
            for q in range(len(id_str)):
                bbox = bbox_[q]
                id = int(id_str[q])
                prob = conf[q]
                dim = (300, 500)

                # Crop the image based on the bounding box coordinates
                print(bbox)
                center_x = int(bbox[0] * width)
                center_y = int(bbox[1] * height)
                w = int(bbox[2] * width)
                h = int(bbox[3] * height)

                    # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cropped_image = image[y:y+h, x:x+w]
                crop = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_NEAREST)
                # Construct the output filename
                output_filename = f"videos/boundingbox/st1in_id{id}_Frame{i}_prob{prob}.jpg"  # Adjust the file extension if necessary
                
                # Save the cropped image
                cv2.imwrite(output_filename, crop)
                
                # Print the status
                print(f"Saved cropped image: {output_filename}")
            else:
                # Print a warning if the image file is missing
                print(f"Image file not found: {image_file_path}")
 