import cv2
import os

def video_to_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 1
    frame_interval = 1  # Read every fifth frame

    while True:
        # Read a single frame from the video
        ret, frame = video.read()

        # Break the loop if no more frames are available
        if not ret:
            print("no more", frame_count)
            break

        # Process and save the frame only if it is the desired frame
        if frame_count % frame_interval == 0:
            # Save the frame as an image file
            frame_path = os.path.join(output_folder, f"frame{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

        # Increment frame count
        frame_count += 1

    # Release the video file and close any open windows
    video.release()
    cv2.destroyAllWindows()

# Specify the path to your video file
video_path = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/videos/station1_in.mp4"

# Specify the output folder to store the frames
output_folder = 'C:/Users/Hanzalah Choudhury/Desktop/boundingbox/videos/data'

# Convert the video into frames and store every fifth frame in the output folder
video_to_frames(video_path, output_folder)