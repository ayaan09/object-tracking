import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_similarity(img1, img2):
    # Convert images to grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM similarity between the grayscale images
    similarity = ssim(gray_img1, gray_img2)
    return similarity

def remove_similar_images(folder_path, similarity_threshold):
    # Get the list of image filenames in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Load and process images
    images = []
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)

    # Create similarity matrix
    num_images = len(images)
    similarity_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i+1, num_images):
            similarity = calculate_similarity(images[i], images[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Check similarity between images and remove similar images
    num_images = len(images)
    keep_images = [True] * num_images
    for i in range(num_images):
        if not keep_images[i]:
            continue

        for j in range(i+1, num_images):
            if not keep_images[j]:
                continue

            if similarity_matrix[i, j] >= similarity_threshold:
                keep_images[j] = False

    # Remove similar images from folder
    for i, keep in enumerate(keep_images):
        if not keep:
            image_path = os.path.join(folder_path, image_files[i])
            print(image_path)

# Specify the folder path containing the images
folder_path = 'C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data'

# Specify the similarity threshold (adjust as needed)
similarity_threshold = 0.6

# Remove similar images from the folder
remove_similar_images(folder_path, similarity_threshold)