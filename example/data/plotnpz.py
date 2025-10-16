import numpy as np
import matplotlib.pyplot as plt

# def plot_npz_image(npz_path):
#     """
#     Load and plot an image stored in a .npz file.
#     """
#     # Load .npz file
#     data = np.load(npz_path)
#     img = data['image']

#     # Plot the image
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f"Image from {npz_path}")
#     plt.show()

# if __name__ == "__main__":
#     npz_file = "binary_example.npz"  # change to your file name
#     plot_npz_image(npz_file)

npz_file = "Euler2original.npz"  # change to your file name
#npz_file = "composite_example.npz"  # change to your file name

data = np.load(npz_file)


"""
Load and plot an image stored in a .npz file.
"""
# Load .npz file
# Get the first key (e.g., "array1")
key = list(data.keys())[0]
img = data[key]

# Plot
plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
plt.axis('off')
plt.title(f"{npz_file} [{key}]")
plt.show()


# flipped = 1 - img  # or np.logical_not(img).astype(np.uint8)
# plt.imshow(flipped, cmap='gray' if img.ndim == 2 else None)
# plt.axis('off')
# plt.title(f"{npz_file} [{key}]")
# plt.show()

#np.savez_compressed(npz_file_new, array1=flipped)


#convert to binary..


def to_binary_image(img, threshold=0.5):
    """
    Convert an RGB(A) or grayscale image array to binary (0 or 1).
    """
    # If image has 3 or 4 channels, convert to grayscale
    if img.ndim == 3:
        img = img[..., :3]  # drop alpha if present
        img = img.mean(axis=-1)  # average R,G,B channels

    # Normalize if necessary (some images are 0–255)
    if img.max() > 1.0:
        img = img / 255.0

    # Threshold to get binary image
    binary_img = (img > threshold).astype(np.uint8)
    return binary_img

binary_img = to_binary_image(img, threshold=0.78)

# Visualize
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_img, cmap='gray')
plt.title("Binary")
plt.axis("off")
plt.show()

npz_file_new = "Euler2original_binary.npz"  # change to your file name
np.savez_compressed(npz_file_new, array1=binary_img)
#np.savez_compressed("metal_example_binary.npz", array1=img)


# data = np.load(npz_file)
# img = data['image']

# # Plot the image
# plt.imshow(img)
# plt.axis('off')
# plt.title(f"Image from {npz_path}")
# plt.show()