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

## Installation

To install the necessary Python packages, run:

```sh
pip install numpy mahotas scikit-image matplotlib
```


## Usage

### Main Functions

#### `generate_realistic_cell(size, fluorescence_level)`

This function generates an oval-shaped yeast cell with slight random variations in its radii to simulate realistic cell shapes.

- **Inputs:**
  - `size`: A tuple `(width, height)` indicating the base size of the cell.
  - `fluorescence_level`: An integer specifying the intensity level of the cell's fluorescence.

- **Output:**
  - Returns a `numpy` array representing the generated cell with the specified fluorescence intensity. The cell is represented as an oval with random radii to introduce natural variability.

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

### Example Script

An example script is provided to demonstrate how to use these functions to generate and visualize synthetic yeast cell images. The script generates a 2048x2048 image populated with 200 randomly sized yeast cells, visualizes the fluorescence image, and displays the labeled image where each cell is uniquely identified.
