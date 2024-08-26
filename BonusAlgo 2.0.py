import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, jaccard_score
from cellpose import models as cellpose_models
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Define a function to evaluate and compare different models
def evaluate_models(image_path, ground_truth_path):
    # Load the images
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Convert the single-channel grayscale image to a 3-channel image
    image_3c = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Placeholder dictionary to store results from different models
    results = {}

    # Ensure the ground truth is binarized
    ground_truth_binary = (ground_truth_mask > 0).astype(np.uint8)

    # Cellpose Model
    print("Running Cellpose...")
    cellpose_model = cellpose_models.Cellpose(model_type='cyto')
    masks, flows, styles, diams = cellpose_model.eval(image, diameter=None, channels=[0, 0])
    masks_resized = cv2.resize(masks.astype(np.uint8), (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    predicted_binary = (masks_resized > 0).astype(np.uint8)
    results['Cellpose'] = evaluate_performance(ground_truth_binary, predicted_binary)

    # U-Net Model (simplified, using torch)
    print("Running U-Net...")
    unet_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)  # Placeholder for actual U-Net
    unet_model.eval()
    with torch.no_grad():
        input_tensor = F.to_tensor(image_3c).unsqueeze(0)  # Convert 3-channel image to tensor
        output = unet_model(input_tensor)['out']
        unet_pred = output.squeeze(0).argmax(0).byte().numpy()
    predicted_binary = (unet_pred > 0).astype(np.uint8)
    results['U-Net'] = evaluate_performance(ground_truth_binary, predicted_binary)

    # Mask R-CNN Model (using Detectron2)
    print("Running Mask R-CNN...")
    maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
    maskrcnn_model.eval()
    with torch.no_grad():
        input_tensor = F.to_tensor(image_3c).unsqueeze(0)  # Use the 3-channel image
        output = maskrcnn_model(input_tensor)
        maskrcnn_pred = output[0]['masks'].squeeze(1).cpu().numpy()
    predicted_binary = (np.sum(maskrcnn_pred, axis=0) > 0.5).astype(np.uint8)
    results['Mask R-CNN'] = evaluate_performance(ground_truth_binary, predicted_binary)

    # Print and visualize results
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        print(f"Binary Precision: {metrics['precision']}")
        print(f"Binary Recall: {metrics['recall']}")
        print(f"IoU: {metrics['iou']}")
        visualize_results(image, metrics['predicted'], ground_truth_binary, model_name)

def evaluate_performance(ground_truth, predicted):
    # Flatten the masks for comparison
    ground_truth_flat = ground_truth.flatten()
    predicted_flat = predicted.flatten()

    # Calculate metrics
    precision = precision_score(ground_truth_flat, predicted_flat, average='binary')
    recall = recall_score(ground_truth_flat, predicted_flat, average='binary')
    iou = jaccard_score(ground_truth_flat, predicted_flat)

    return {
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'predicted': predicted
    }

def visualize_results(image, predicted_mask, ground_truth_mask, model_name):
    overlay = cv2.addWeighted(ground_truth_mask * 255, 0.5, predicted_mask * 255, 0.5, 0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'{model_name} - Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='jet')
    plt.title(f'{model_name} - Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay, cmap='gray')
    plt.title(f'{model_name} - Overlay of Ground Truth and Prediction')
    plt.axis('off')

    plt.show()

# Run the evaluation for multiple models
    evaluate_models('output_images/fluorescence_image.png', 'mask_images/labelled_image.png')