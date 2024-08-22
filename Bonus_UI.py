import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
    cell_properties = [{'size': (np.random.randint(low_size_cell, high_size_cell), np.random.randint(low_size_cell, high_size_cell)),
                        'fluorescence_level': np.random.randint(fluorescence_level_min, fluorescence_level_max)}
                        for _ in range(num_cells)]

    # Generate images
    fluorescence_image, labelled_image = place_realistic_cells(image_size, num_cells, cell_properties)

    # Display images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cmap_fluorescence = mcolors.LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])
    ax[0].imshow(fluorescence_image, cmap=cmap_fluorescence)
    ax[0].set_title('Fluorescence Image')
    ax[1].imshow(labelled_image, cmap=plt.cm.jet)
    ax[1].set_title('Labeled Image')

    for widget in image_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Create the main window
root = tk.Tk()
root.title("Yeast Cell Image Generator")

# UI elements for image size
ttk.Label(root, text="Image Size:").grid(column=0, row=0, padx=5, pady=5)
img_size_entry = ttk.Entry(root)
img_size_entry.insert(0, "2048")
img_size_entry.grid(column=1, row=0, padx=5, pady=5)

# UI elements for number of cells
ttk.Label(root, text="Number of Cells:").grid(column=0, row=1, padx=5, pady=5)
num_cells_entry = ttk.Entry(root)
num_cells_entry.insert(0, "50")
num_cells_entry.grid(column=1, row=1, padx=5, pady=5)

# UI elements for cell size range
ttk.Label(root, text="Min Cell Size:").grid(column=0, row=2, padx=5, pady=5)
low_size_entry = ttk.Entry(root)
low_size_entry.insert(0, "10")
low_size_entry.grid(column=1, row=2, padx=5, pady=5)

ttk.Label(root, text="Max Cell Size:").grid(column=0, row=3, padx=5, pady=5)
high_size_entry = ttk.Entry(root)
high_size_entry.insert(0, "20")
high_size_entry.grid(column=1, row=3, padx=5, pady=5)

# UI elements for fluorescence level
ttk.Label(root, text="Min Fluorescence Level:").grid(column=0, row=4, padx=5, pady=5)
fluorescence_min_entry = ttk.Entry(root)
fluorescence_min_entry.insert(0, "1000")
fluorescence_min_entry.grid(column=1, row=4, padx=5, pady=5)

ttk.Label(root, text="Max Fluorescence Level:").grid(column=0, row=5, padx=5, pady=5)
fluorescence_max_entry = ttk.Entry(root)
fluorescence_max_entry.insert(0, "65535")
fluorescence_max_entry.grid(column=1, row=5, padx=5, pady=5)

# Generate button
generate_button = ttk.Button(root, text="Generate & Preview", command=generate_and_preview)
generate_button.grid(column=0, row=6, columnspan=2, padx=5, pady=10)

# Frame for displaying the image
image_frame = ttk.Frame(root)
image_frame.grid(column=2, row=0, rowspan=6, padx=10, pady=10)

# Run the UI loop
root.mainloop()
