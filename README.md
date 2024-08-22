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
