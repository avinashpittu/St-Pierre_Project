import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Load the ground truth mask
ground_truth_mask = cv2.imread('mask_images/labelled_image.png', cv2.IMREAD_GRAYSCALE)

# Load pre-trained Mask R-CNN model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"  # Use CPU

predictor = DefaultPredictor(cfg)

# Load the image and convert it to RGB
fluorescence_image_rgb = cv2.imread("output_images/fluorescence_image.png")
fluorescence_image_rgb = cv2.cvtColor(fluorescence_image_rgb, cv2.COLOR_BGR2RGB)

# Run the predictor
outputs = predictor(fluorescence_image_rgb)

# Visualize the results
v = Visualizer(fluorescence_image_rgb[:, :, ::-1], scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Show the segmentation result
cv2.imshow("Segmentation", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the colored segmentation
colored_segmentation = cv2.applyColorMap( out.get_image()[:, :, ::-1], cv2.COLORMAP_JET)
cv2.imshow("Colored Segmentation", colored_segmentation)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Convert the outputs to a binary mask (for instance segmentation)
# Combine masks from all instances
predicted_mask = np.max(outputs["instances"].pred_masks.to("cpu").numpy(), axis=0).astype(np.uint8) * 255

# Ensure that predicted_mask has the same shape as ground_truth_mask
predicted_mask_resized = cv2.resize(predicted_mask, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

# Threshold the predicted mask to ensure it's binary
predicted_mask_binary = (predicted_mask_resized > 0.5).astype(np.uint8) * 255

# Flatten the masks to turn them into 1D arrays
ground_truth_flat = ground_truth_mask.flatten()
predicted_flat = predicted_mask_binary.flatten()

# Ensure binary masks: Convert all non-zero values to 1, making it binary
ground_truth_flat_binary = (ground_truth_flat > 0).astype(int)
predicted_flat_binary = (predicted_flat > 0).astype(int)

# Calculate precision and recall as binary
precision = precision_score(ground_truth_flat_binary, predicted_flat_binary, average='binary')
recall = recall_score(ground_truth_flat_binary, predicted_flat_binary, average='binary')

print(f'Binary Precision: {precision}')
print(f'Binary Recall: {recall}')

# Visualize the differences between ground truth and prediction
difference = cv2.absdiff(ground_truth_mask, predicted_mask_binary)
cv2.imshow("Difference", difference)
cv2.waitKey(0)
cv2.destroyAllWindows()


