# St-Pierre_Project
Project to create a synthetic image generator  to generate image and labeled image pairs of yeast cells under fluorescence microscopy.

## Overview

This repository contains code to generate synthetic images of yeast cells under fluorescence microscopy. The cells are modeled as ovals with slight variations to simulate realistic shapes. The output includes both fluorescence images and corresponding labeled images.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `mahotas`
  - `scikit-image`
  - `matplotlib`


## Usage

### Example Script

To demonstrate how to use the synthetic image generator functions, an exemplar script is provided. This script showcases how to generate a fluorescence image and a labeled image using the `place_realistic_cells` function.

#### Usage Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/avinashpittu/St-Pierre_Project.git

### Install the Required Dependencies:

```bash
pip install numpy mahotas scikit-image matplotlib
```

### Run the Exemplar Script:
```bash
python example_script.py
```

### Expected Output:
The script generates and displays:

  - A Fluorescence Image: A grayscale image showing synthetic yeast cells.
  - A Labeled Image: A color-labeled image where each yeast cell is marked with a unique identifier.


#### `place_realistic_cells(image_size, num_cells, cell_properties)`

This function creates a synthetic fluorescence microscopy image populated with multiple yeast cells, along with a labeled image where each cell is uniquely identified.

- **Inputs:**
  - `image_size`: A tuple `(height, width)` specifying the dimensions of the output image.
  - `num_cells`: An integer indicating the number of cells to generate and place within the image.
  - `cell_properties`: A list of dictionaries where each dictionary contains:
    - `size`: A tuple defining the size of the cell.
    - `fluorescence_level`: The fluorescence intensity of the cell.

- **Outputs:**
  - `fluorescence_image`: A `numpy` array representing the generated fluorescence image, where each pixel corresponds to the fluorescence intensity of the cells.
  - `labelled_image`: A `numpy` array of the same size as `fluorescence_image`, where each cell is assigned a unique integer label.
