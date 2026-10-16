import cv2
import numpy as np
import matplotlib.pyplot as plt
from pacbot.variables import *



# NOTE: These values are set based on the specific video and may need to 
# be adjusted for different videos, different colored blocks, different shape
# blocks, etc.
min_brightness_thresh = 240 # minimum brightness threshold for binarization
min_area = 2000.0 # minimum area of connected components to keep (in pixels)
ghost_home_direction = left # direction of the ghost home in the video (left, right, up, down)
black_rgb = (240, 210, 133) # set this to the color of the blocks 
white_rgb = (230, 230, 230) # play around with this value. should be between black_rgb and 255
height_adjustment_ratio = 660.0/620.0 # adjusts the height of the bounds to account for the inner south walls not extending all the way to the outer walls

def normalize_rotation_to_grid(image, ghost_home_direction=ghost_home_direction):
    """
    Normalizes rotation of the image to align with the grid from pacbot.grid.grid
    """
    match ghost_home_direction:
        case x if x == left:
            return np.rot90(image, k=2)  # Rotate 180 degrees
        case x if x == right:
            return image # No rotation needed
        case x if x == up:
            return np.rot90(image, k=1)  # Rotate 90 degrees counter-clockwise
        case x if x == down:
            return np.rot90(image, k=3)  # Rotate 270 degrees counter-clockwise



def convert_to_grayscale(image, black_rgb=black_rgb, white_rgb=white_rgb):
    """
    Converts an RGB image to grayscale based on the provided black and 
    white reference colors. 
    """

    # convert the tuples to np arrays
    c_black = np.array(black_rgb, dtype=np.float32)
    c_white = np.array(white_rgb, dtype=np.float32)

    # Compute the axis vector from black to white
    axis = c_white - c_black
    axis_norm_sq = np.dot(axis, axis)

    # Compute the projection of each pixel onto the axis
    diff = image - c_black
    # Compute the projection scalar t for each pixel
    t = np.sum(diff * axis, axis=-1) / axis_norm_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    # Scale to 0-255 and convert to uint8
    gray_array = (t_clamped * 255.0).astype(np.uint8)
    return gray_array


def display_image(image, window_name="Image", fignum = None):
    """
    Displays an image using matplotlib. If fignum is provided, 
    it will use that figure number; otherwise, it will create a new figure.
    """
    plt.ion()
    if (fignum is not None):
        plt.figure(fignum)
    else:
        plt.figure()
    plt.clf()
    plt.imshow(image, cmap="gray")
    plt.title(window_name)
    plt.axis("off")
    plt.show()

def clean(image, min_area=min_area):
    """
    Cleans the binary image by removing small, unconnected components 
    based on the specified minimum area.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    clean_mask = np.zeros_like(image)
    for i in range(1, num_labels):  # Skip index 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask

def adjust_bounds_height(bounds, ghost_home_direction=ghost_home_direction):
    """
    Adjusts the height of the bounds based on the direction of the ghost home.
    This is necessary because the inner south walls do not extend all the way 
    to the outer walls.
    """
    x_min, x_max, y_min, y_max = bounds
    if (ghost_home_direction == up):
        height_adjusted = (y_max - y_min) * (height_adjustment_ratio)
        y_min = y_max - height_adjusted
    elif ghost_home_direction == down:
        height_adjusted = (y_max - y_min) * (height_adjustment_ratio)
        y_max = y_min + height_adjusted
    elif ghost_home_direction == left:
        height_adjusted = (x_max - x_min) * (height_adjustment_ratio)
        x_max = x_min + height_adjusted
    elif ghost_home_direction == right:
        height_adjusted = (x_max - x_min) * (height_adjustment_ratio)
        x_min = x_max - height_adjusted
    return (x_min, x_max, y_min, y_max)

def display_bounds(image, bounds, fignum = None):
    """
    Displays the image with the specified bounds overlaid as a rectangle.
    """
    x_min, x_max, y_min, y_max = bounds
    plt.ion()
    if (fignum is not None):
        plt.figure(fignum)
    else:
        plt.figure()
    plt.clf()
    plt.imshow(image, origin = 'upper')
    plt.title("Image with Bounds")
    plt.axis("off")
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      edgecolor='red', facecolor='none', linewidth=2))
    plt.show()

def extract_bounds(image,
                   min_brightness_thresh=min_brightness_thresh, 
                   min_area=min_area, 
                   ghost_home_direction=ghost_home_direction,
                   debug_display=False,
                   decimation = 1):
    """
    Extracts the bounds of the map from the given image.
    """

    # Decimate the image for faster processing
    decimated = image[::decimation, ::decimation]
    # Rotate the image to align with the grid
    rotated_image = normalize_rotation_to_grid(decimated, ghost_home_direction=ghost_home_direction)
    # convert to binary
    gray_image = convert_to_grayscale(rotated_image)
    thresh, binary_image = cv2.threshold(gray_image, min_brightness_thresh, 255, cv2.THRESH_BINARY)

    # Clean the binary image to remove small components
    clean_image = clean(binary_image, min_area=min_area/decimation**2) # decimation squared bc area scales with square of decimation

    # Find the bounds of the clean image
    ys, xs = np.where(clean_image == 255)
    bounds = (np.min(xs), np.max(xs), np.min(ys), np.max(ys))  

    # Adjust the bounds height to account for the inner south walls not extending all the way to the outer walls
    bounds = adjust_bounds_height(bounds, ghost_home_direction=ghost_home_direction)

    # Debug display of intermediate steps
    if debug_display:
        display_image(image, "Raw Image", fignum=1)
        display_bounds(rotated_image, (np.min(xs),np.max(xs), np.min(ys), np.max(ys)), fignum=2)
        display_image(gray_image, "Grayscale Image", fignum=3)
        display_image(binary_image, "Binary Image", fignum=4)
        display_image(clean_image, "Clean Image", fignum=5)

    # Return the bounds scaled back to the original image size
    return bounds[0]*decimation, bounds[1]*decimation, bounds[2]*decimation, bounds[3]*decimation