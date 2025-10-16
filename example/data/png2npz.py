import numpy as np
import matplotlib.pyplot as plt
import os

def png_to_npz(png_path, output_path=None):
    """
    Convert a .png image to a .npz file without using PIL.
    """
    # Read image using matplotlib
    img_array = plt.imread(png_path)

    # Determine output filename
    if output_path is None:
        base = os.path.splitext(png_path)[0]
        output_path = base + '.npz'

    # Save as compressed npz
    np.savez_compressed(output_path, image=img_array)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_png = "Euler2original.jpg"  # change to your image
    png_to_npz(input_png)
