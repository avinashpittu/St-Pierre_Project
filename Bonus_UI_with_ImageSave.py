import os
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

# Import your SyntheticImageGenerator function (ensure this is correctly implemented)
from SyntheticImageGenerator import place_realistic_cells


def generate_and_preview():
    # Get values from UI
    image_size = (int(img_size_entry.get()), int(img_size_entry.get()))
    num_cells = int(num_cells_entry.get())
    low_size_cell = int(low_size_entry.get())
    high_size_cell = int(high_size_entry.get())
    fluorescence_level_min = int(fluorescence_min_entry.get())
    fluorescence_level_max = int(fluorescence_max_entry.get())

    # Define cell properties
    cell_properties = [{'size': (np.random.randint(low_size_cell, high_size_cell),
                                 np.random.randint(low_size_cell, high_size_cell)),
                        'fluorescence_level': np.random.randint(fluorescence_level_min, fluorescence_level_max)}
                       for _ in range(num_cells)]

    # Generate images using your custom function
    fluorescence_image, labelled_image = place_realistic_cells(image_size, num_cells, cell_properties)

    # Save images to the 'output_images' and 'mask_images' folders
    output_dir = "output_images"
    mask_dir = "mask_images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Paths for saving images
    fluorescence_image_path = os.path.join(output_dir, "fluorescence_image.png")
    labelled_image_path = os.path.join(mask_dir, "labelled_image.png")


    # Convert grayscale images to RGB
    cmap_fluorescence = mcolors.LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])
    # Normalize the fluorescence image between 0 and 1 before applying colormap
    norm_fluorescence_image = (fluorescence_image - fluorescence_image.min()) / (
                fluorescence_image.max() - fluorescence_image.min())
    fluorescence_image_rgb = (cmap_fluorescence(norm_fluorescence_image)[:, :, :3] * 255).astype(np.uint8)
    cmap_label = plt.cm.get_cmap('jet')
    # Normalize the labeled image similarly
    norm_labelled_image = (labelled_image - labelled_image.min()) / (labelled_image.max() - labelled_image.min())
    labelled_image_rgb = (cmap_label(norm_labelled_image)[:, :, :3] * 255).astype(np.uint8)


    # Convert numpy arrays to images and save
    fluorescence_img = Image.fromarray(fluorescence_image_rgb)
    fluorescence_img.save(fluorescence_image_path)

    labelled_img = Image.fromarray(labelled_image_rgb)
    labelled_img.save(labelled_image_path)

    print(f"Fluorescence Image saved to {fluorescence_image_path}")
    print(f"Labeled Image saved to {labelled_image_path}")

    # Display images in the UI
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(fluorescence_image_rgb)
    ax[0].set_title('Fluorescence Image')
    ax[1].imshow(labelled_image_rgb)
    ax[1].set_title('Labeled Image')

    # Clear previous images from the UI frame
    for widget in image_frame.winfo_children():
        widget.destroy()

    # Add the new canvas to display the images
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Create the main window
root = tk.Tk()
root.title("Yeast Cell Image Generator")

# UI elements for image size
ttk.Label(root, text="Image Size:").grid(column=0, row=0, padx=5, pady=5)
img_size_entry = ttk.Entry(root)
img_size_entry.insert(0, "2048")  # default value
img_size_entry.grid(column=1, row=0, padx=5, pady=5)

# UI elements for number of cells
ttk.Label(root, text="Number of Cells:").grid(column=0, row=1, padx=5, pady=5)
num_cells_entry = ttk.Entry(root)
num_cells_entry.insert(0, "50")  # default value
num_cells_entry.grid(column=1, row=1, padx=5, pady=5)

# UI elements for cell size range
ttk.Label(root, text="Min Cell Size:").grid(column=0, row=2, padx=5, pady=5)
low_size_entry = ttk.Entry(root)
low_size_entry.insert(0, "10")  # default value
low_size_entry.grid(column=1, row=2, padx=5, pady=5)

ttk.Label(root, text="Max Cell Size:").grid(column=0, row=3, padx=5, pady=5)
high_size_entry = ttk.Entry(root)
high_size_entry.insert(0, "20")  # default value
high_size_entry.grid(column=1, row=3, padx=5, pady=5)

# UI elements for fluorescence level
ttk.Label(root, text="Min Fluorescence Level:").grid(column=0, row=4, padx=5, pady=5)
fluorescence_min_entry = ttk.Entry(root)
fluorescence_min_entry.insert(0, "1000")  # default value
fluorescence_min_entry.grid(column=1, row=4, padx=5, pady=5)

ttk.Label(root, text="Max Fluorescence Level:").grid(column=0, row=5, padx=5, pady=5)
fluorescence_max_entry = ttk.Entry(root)
fluorescence_max_entry.insert(0, "65535")  # default value
fluorescence_max_entry.grid(column=1, row=5, padx=5, pady=5)

# Generate button
generate_button = ttk.Button(root, text="Generate & Preview", command=generate_and_preview)
generate_button.grid(column=0, row=6, columnspan=2, padx=5, pady=10)

# Frame for displaying the image
image_frame = ttk.Frame(root)
image_frame.grid(column=2, row=0, rowspan=6, padx=10, pady=10)

# Run the UI loop
root.mainloop()