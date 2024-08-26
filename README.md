# Synthetic Image Generator

This project generates synthetic fluorescence microscopy images of yeast cells with realistic properties such as cell size, fluorescence intensity, and overlapping cells. The code allows users to simulate images for testing and validation of image processing algorithms.

## Features

- **Realistic Cell Simulation**: Generates oval-shaped yeast cells with randomized sizes and fluorescence levels.
- **Controlled Overlap**: Allows for controlled overlap between cells, simulating crowded environments.
- **Adjustable Noise**: Adds Gaussian noise to the image for realistic noise simulation.

### Requirements

- Python 3.7 or higher
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-image`
  - `mahotas`

## Usage

## Project Structure

  - SyntheticImageGenerator.py: Contains the core function place_realistic_cells for generating the synthetic images.
  - example_script.py: An example script to demonstrate how to use the code.
  - README.md: Documentation for the project.

### Expected Output:
The script generates and displays:

  - A Fluorescence Image: A grayscale image showing synthetic yeast cells.
  - A Labeled Image: A color-labeled image where each yeast cell is marked with a unique identifier.

These images are displayed side by side for easy comparison.

### Example Script

To demonstrate how to use the synthetic image generator functions, an exemplar script is provided. This script showcases how to generate a fluorescence image and a labeled image using the `place_realistic_cells` function.

#### Usage Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/avinashpittu/St-Pierre_Project.git

### Install the Required Dependencies:

If you are in python environment like pycharm
```bash
pip install numpy mahotas scikit-image matplotlib
```
or if you are in command line 
```bash
python -m pip install numpy mahotas scikit-image matplotlib
```

### Run the Exemplar Script:

```bash
python example_script.py
```

### The script will generate and display the synthetic images.


Additional Observations


#### `place_realistic_cells(image_size, num_cells, cell_properties)`

This function creates a synthetic fluorescence microscopy image populated with multiple yeast cells, along with a labeled image where each cell is uniquely identified.

- **Inputs:** The script allows for customization of cell properties, such as size and fluorescence intensity, making it versatile for different simulation needs.
  
  - `image_size`: A tuple `(height, width)` specifying the dimensions of the output image.
  - `num_cells`: An integer indicating the number of cells to generate and place within the image.
  - `cell_properties`: A list of dictionaries where each dictionary contains:
    - `size`: A tuple defining the size of the cell.
    - `fluorescence_level`: The fluorescence intensity of the cell.
      
- The controlled overlap feature is useful for generating realistic crowded environments often observed in biological samples.
- The adjustable noise feature simulates the real-world conditions of microscopy images, which typically contain some level of noise.

- **Outputs:**
  - `fluorescence_image`: A `numpy` array representing the generated fluorescence image, where each pixel corresponds to the fluorescence intensity of the cells.
  - `labelled_image`: A `numpy` array of the same size as `fluorescence_image`, where each cell is assigned a unique integer label.

## References

- [scikit-image: Image processing in Python](https://scikit-image.org/)
- [NumPy: The fundamental package for scientific computing with Python](https://numpy.org/)
- [Matplotlib: Visualization with Python](https://matplotlib.org/)

### Note
  - You can see the example outputs in "SyntheticImageGenerator.ipynb". I ran that code in Jupyter Notebook to show it as an example


# Bonus(UI)

## Yeast Cell Image Generator with UI

This Tkinter application allows users to generate and preview images of yeast cells with realistic properties. The user can specify various parameters such as image size, number of cells, cell size range, and fluorescence levels. The generated images include a fluorescence image and a labeled image.

## Features

- Generate fluorescence and labeled images of yeast cells.
- Customize image size, number of cells, cell size range, and fluorescence levels.
- Preview the generated images within the application.
- Exit the application with a single button click.


### Install the Required Dependencies:

```bash
pip install numpy tkinter
```

### Run the Exemplar Script:

```bash
python Bonus_UI.py
```

### The script will generate and display the synthetic images.


Additional Observations


#### `place_realistic_cells(image_size, num_cells, cell_properties)`

This function creates a synthetic fluorescence microscopy image populated with multiple yeast cells, along with a labeled image where each cell is uniquely identified.

- **Inputs:** The script allows for customization of cell properties, such as size and fluorescence intensity, making it versatile for different simulation needs.
  
  - `image_size`: A tuple `(height, width)` specifying the dimensions of the output image.
  - `num_cells`: An integer indicating the number of cells to generate and place within the image.
  - `cell_properties`: A list of dictionaries where each dictionary contains:
    - `size`: A tuple defining the size of the cell.
    - `fluorescence_level`: The fluorescence intensity of the cell.
      
- The controlled overlap feature is useful for generating realistic crowded environments often observed in biological samples.
- The adjustable noise feature simulates the real-world conditions of microscopy images, which typically contain some level of noise.

- **Outputs:**
  - `fluorescence_image`: A `numpy` array representing the generated fluorescence image, where each pixel corresponds to the fluorescence intensity of the cells.
  - `labelled_image`: A `numpy` array of the same size as `fluorescence_image`, where each cell is assigned a unique integer label.

 ### Note
  - You can see the example outputs in "SyntheticImageGenerator.ipynb". I ran that code in Jupyter Notebook to show it as an example




### Note
  - Run `Bonus_UI_with_ImageSave.py` before running `BonusAlgo 2.0.py` to save the images that generated from the UI to repective directories `output_images/fluorescence_image.png`
`mask_images/labelled_image.png`. 





# BonusAlgo 2.0.py

## Introduction

The `BonusAlgo 2.0.py` script not only utilizes the Mask R-CNN model but also integrates the Cellpose model to segment and analyze synthetic fluorescence microscopy images. This script aims to compare the performance of different segmentation models on the same dataset and provide a more comprehensive evaluation.

## Prerequisites

Before running this script, ensure you have Python 3.6 or later and the following dependencies installed:

```bash
pip install opencv-python numpy matplotlib scikit-learn cellpose torch torchvision
```

### Additional Dependencies

- **Cellpose**: A deep learning-based model specifically designed for segmenting biological images.
- **PyTorch**: Required for running the Mask R-CNN model.
- **Torchvision**: Provides various utilities and pre-trained models for PyTorch.

## How to Run the Script

### Step 1: Prepare the Required Images

Make sure the following images are available in the specified directories:
- **Ground Truth Mask:** The labeled image (`labelled_image.png`) should be in the `mask_images/` directory.
- **Fluorescence Image:** The synthetic image (`fluorescence_image.png`) should be located in the `output_images/` directory.

These images can be generated using the `Bonus_UI_with_ImageSave.py` script.

### Step 2: Execute the Script

Run the script using the following command:

```bash
python BonusAlgo 2.0.py
```

### Step 3: Analyze the Outputs

The script will compare the segmentation results from the Mask R-CNN and Cellpose models and provide various metrics to assess their performance.

#### Outputs:
- **Precision, Recall, and Jaccard Index**: The script calculates these metrics for both models and prints them to the console.
- **Visualized Results**: Visualizations of the segmented images produced by both models, allowing for a direct comparison of their performance.

## Code Overview

### Key Components:

- **Model Initialization**:
  The script initializes both the Mask R-CNN model from `torchvision` and the Cellpose model. The Mask R-CNN is pre-trained on the COCO dataset, while the Cellpose model is specialized for biological image segmentation.

  ```python
  maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
  cellpose_model = cellpose_models.Cellpose(gpu=False, model_type='cyto')
  ```

- **Image Loading and Preprocessing**:
  The synthetic fluorescence image is loaded in grayscale, then converted to a 3-channel image required by the Mask R-CNN model.

  ```python
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  image_3c = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  ```

- **Model Evaluation**:
  The script applies both the Mask R-CNN and Cellpose models to the loaded image, then evaluates their performance against the ground truth mask.

  ```python
  results['Mask R-CNN'] = maskrcnn_model(image_3c)
  results['Cellpose'] = cellpose_model.eval(image)
  ```

- **Performance Metrics**:
  Precision, recall, and the Jaccard index are calculated for both models to provide a quantitative assessment of their performance.

  ```python
  precision = precision_score(ground_truth_binary.flatten(), prediction_binary.flatten())
  recall = recall_score(ground_truth_binary.flatten(), prediction_binary.flatten())
  jaccard = jaccard_score(ground_truth_binary.flatten(), prediction_binary.flatten())
  ```

## Conclusion

The `BonusAlgo 2.0.py` script functionlity is incorporating an additional model (Cellpose) specifically designed for biological image segmentation. This script provides a comprehensive comparison of two different segmentation approaches, offering insights into which model performs better on synthetic fluorescence images. The outputs, including precision, recall, and Jaccard index, help quantify the effectiveness of each model.

Ensure that the required dependencies are installed, and the correct images are in place before running the script. This script is a powerful tool for evaluating and comparing segmentation models on synthetic datasets.


## Note
  - Also, tried Detectron2 since none of the algo above mentioned didn't give desired results. 



# Bonus_AlgoTest.py

## Introduction

The `Bonus_AlgoTest.py` script is designed to apply a pre-trained Mask R-CNN model from Detectron2 to segment and analyze synthetic fluorescence microscopy images. This script evaluates the performance of the model by comparing its predictions to a provided ground truth mask, using metrics like precision and recall.

## Prerequisites

Before running the script, ensure that you have Python 3.6 or later installed. You will also need the following dependencies:

```bash
pip install detectron2 opencv-python numpy scikit-learn
```

`detectron2` is a Facebook AI Research (FAIR) library that requires specific installation instructions depending on your environment. You can find detailed instructions on the official [Detectron2 GitHub repository](https://github.com/facebookresearch/detectron2).

## How to Run the Script

### Step 1: Prepare the Required Images

The script expects the following images to be available in the specified directories:
- **Ground Truth Mask:** A labeled image named `labelled_image.png` should be placed in the `mask_images/` directory.
- **Fluorescence Image:** The synthetic fluorescence image named `fluorescence_image.png` should be located in the `output_images/` directory.

These images can be generated using the `Bonus_UI_with_ImageSave.py` script.

### Step 2: Execute the Script

Run the script with the following command:

```bash
python Bonus_AlgoTest.py
```

### Step 3: Analyze the Outputs

The script processes the input fluorescence image and generates predictions using a pre-trained Mask R-CNN model. It then compares these predictions against the ground truth mask to evaluate the model's performance.

#### Outputs:
- **Precision and Recall:** The script calculates and prints the precision and recall metrics, which provide insights into the accuracy of the model's predictions.
- **Visualized Output:** The segmented image produced by the Mask R-CNN model is displayed, showcasing the areas identified by the model.

## Code Overview

### Key Components:

- **Detectron2 Configuration:**
  The script configures the Mask R-CNN model using the default settings provided by the Detectron2 model zoo. The configuration includes setting the model's threshold and specifying the use of CPU for computation.

  ```python
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  cfg.MODEL.DEVICE = "cpu"  # Use CPU
  ```

- **Loading and Processing Images:**
  The fluorescence image is loaded and converted to RGB format, which is required by the model.

  ```python
  fluorescence_image_rgb = cv2.imread("output_images/fluorescence_image.png")
  fluorescence_image_rgb = cv2.cvtColor(fluorescence_image_rgb, cv2.COLOR_BGR2RGB)
  ```

- **Model Prediction:**
  The script uses the `predictor` object from Detectron2 to generate predictions on the loaded image.

  ```python
  outputs = predictor(fluorescence_image_rgb)
  ```

- **Performance Evaluation:**
  After generating predictions, the script compares them with the ground truth mask and calculates precision and recall scores.

  ```python
  precision = precision_score(ground_truth_binary.flatten(), prediction_binary.flatten())
  recall = recall_score(ground_truth_binary.flatten(), prediction_binary.flatten())
  ```

## Conclusion

The `Bonus_AlgoTest.py` script provides a straightforward way to evaluate the performance of a pre-trained Mask R-CNN model on synthetic fluorescence images. By calculating metrics such as precision and recall, you can assess how well the model segments cell-like structures in these images.

Ensure that the required dependencies are installed and that the correct images are in place before running the script. The outputs will give you a clear indication of the model's performance, and you can use this information to make further improvements if necessary.


### Note on Results

While the current implementation provides a functional approach to image segmentation, the results from this script may not fully meet expectations in terms of accuracy and precision. The metrics might indicate that the model's performance is suboptimal in this context. However, given additional time, I am confident that I can refine the implementation and improve the results significantly. This could involve fine-tuning the model parameters, experimenting with different pre-trained models, or adjusting the preprocessing steps to better suit the characteristics of the synthetic images.

I value understanding the problem and the implementation more than the immediate performance scores. Therefore, I am committed to improving this script to achieve better results, given the opportunity.





