from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage

# Here you should put the path of your image
IMAGE_PATH = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data/bb/frame120_bb_2.jpg"
Output_path = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data/fashion_new"
cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

def fix_channels(t):
    """
    Some images may have 4 channels (transparent images) or just 1 channel (black and white images), in order to let the images have only 3 channels. I am going to remove the fourth channel in transparent images and stack the single channel in back and white images.
    :param t: Tensor-like image
    :return: Tensor-like image with three channels
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

def idx_to_text(i):
    return cats[i]

# Random colors used for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        ax.text(xmin, ymin, idx_to_text(cl), fontsize=10,
                bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    plt.show()
    plt.savefig("image.png")

def visualize_predictions(image, outputs, threshold=0.8):
        # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    return probas[keep], bboxes_scaled

# def crop_and_save_bounding_boxes(image, probas, bboxes_scaled):
#     for i, (bbox, proba) in enumerate(zip(bboxes_scaled, probas)):
#         xmin, ymin, xmax, ymax = bbox.int().tolist()
#         label = idx_to_text(proba.argmax())
#         # print(label,"CC")
#         # print(proba.tolist())
#         for j, p in enumerate(proba.tolist()):
#             cropped_image = image.crop((xmin, ymin, xmax, ymax))
#             cropped_image.save(f"bbox_{i}_label_{label}_prob_{p:.2f}_{j}.jpg")

def crop_and_save_bounding_boxes(image_path, image_name, probas, bboxes_scaled):
    for i, (bbox, proba) in enumerate(zip(bboxes_scaled, probas)):
        xmin, ymin, xmax, ymax = bbox.int().tolist()
        label = idx_to_text(proba.argmax())
        max_proba, max_index = proba.max(dim=0)
        cropped_image = Image.open(image_path).crop((xmin, ymin, xmax, ymax))
        cropped_image.save(f"{Output_path}/{image_name}_bbox_{i}_label_{label}_prob_{max_proba:.2f}_{max_index}.jpg")

MODEL_NAME = "valentinafeve/yolos-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)




# import os
# folder_path = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data/bb"

# # Iterate over each image file in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         image_path = os.path.join(folder_path, filename)
#         image_name = os.path.splitext(filename)[0]
#         # Load the image and perform bounding box cropping and saving
#         image = Image.open(open(image_path, "rb"))
#         image = fix_channels(ToTensor()(image))
#         # image = image.resize((600, 800))
#         inputs = feature_extractor(images=image, return_tensors="pt")
#         outputs = model(**inputs)
#         probas, bboxes_scaled = visualize_predictions(image, outputs, threshold=0.6)
#         crop_and_save_bounding_boxes(image_path,image_name, probas, bboxes_scaled)

import os
folder_path = "C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data/bb"

# Empty list to store scores
scores = []

# Iterate over each image file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image_name = os.path.splitext(filename)[0]
        # Load the image and perform bounding box cropping and saving
        image = Image.open(open(image_path, "rb"))
        image = fix_channels(ToTensor()(image))
        # image = image.resize((600, 800))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        probas, bboxes_scaled = visualize_predictions(image, outputs, threshold=0.6)
        crop_and_save_bounding_boxes(image_path, image_name, probas, bboxes_scaled)
        num_elements = len(bboxes_scaled)
        scores.append(num_elements)

# Write image names and scores to a text file
output_file = "scores.txt"
with open(output_file, "w") as file:
    for filename, score in zip(os.listdir(folder_path), scores):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_name = os.path.splitext(filename)[0]
            file.write(f"{image_name}: {score}\n")