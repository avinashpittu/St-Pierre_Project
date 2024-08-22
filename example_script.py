import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from SyntheticImageGenerator import place_realistic_cells

# Define image size and number of cells and other parameters
image_size = (2048, 2048)
num_cells = 50
low_size_cell, high_size_cell = 10,20
fluorescence_level_min, fluorescence_level_max = 1000, 65535

# Define cell properties
cell_properties = [{'size': (np.random.randint(low_size_cell, high_size_cell), np.random.randint(low_size_cell, high_size_cell)),
                    'fluorescence_level': np.random.randint(fluorescence_level_min, fluorescence_level_max)} for _ in range(num_cells)]

# Generate the fluorescence image and corresponding labeled image
fluorescence_image, labelled_image = place_realistic_cells(image_size, num_cells, cell_properties)

# Display the generated images
cmap_fluorescence = mcolors.LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
cax1 = ax[0].imshow(fluorescence_image, cmap=cmap_fluorescence)
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_title('Fluorescence Image')

# Display the labeled image with a scale and black background
cax2 = ax[1].imshow(labelled_image, cmap=plt.cm.jet, interpolation='nearest')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_title('Labeled Image')

plt.show()