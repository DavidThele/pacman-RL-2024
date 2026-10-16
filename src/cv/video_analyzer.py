import cv2
from glob import glob
from cv.map_bounds_extractor import extract_bounds
from cv.map_bounds_extractor import display_bounds
from cv.map_bounds_extractor import normalize_rotation_to_grid
from pacbot.grid import grid
from cv.pacbot_tag_finder import find_tag
import matplotlib.pyplot as plt
import numpy as np
from pacbot.variables import *
from tqdm import tqdm


# --- Load video frames from the specified directory ---
videos = glob("../data/*.mp4") 
cap = cv2.VideoCapture(videos[1]) # select the video file to process (currently the first one in the list)

frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video reached or error

    frames.append(frame)
cap.release()

plt.ion()

def draw_pacman_at(position, marker_size=10, direction_length=10, color=(1, 1, 0), rotation = 0):
    """
    Draws a representation of Pacman at the specified position on the current plot.
    The position is given in grid coordinates, and the rotation is in radians."""
    x, y = position
    plt.plot(x, y, 'o', markersize=marker_size, color=color)
    # Draw a line to indicate direction
    end_x = x + direction_length * np.cos(rotation)
    end_y = y + direction_length * np.sin(rotation)
    plt.plot([x, end_x], [y, end_y], color=(0, 0, 0), linewidth=2)

def draw_map_from_grid():
    """
    draws the map based on the grid representation. Each cell 
    in the grid is represented by a different marker and color 
    based on its type. 
    """
    plt.axis("off")
    plt.gca().invert_yaxis()  
    gridw, gridh = len(grid[0]), len(grid)
    for y in range(gridh):
        for x in range(gridw):
            cell = grid[y][x]
            if cell == I:
                plt.plot(x, y, 's', markersize=10, color='black')
            elif cell == o:
                plt.plot(x, y, 'o', markersize=5, color='white')
            elif cell == O:
                plt.plot(x, y, 'o', markersize=8, color='white')
            elif cell == c:
                plt.plot(x, y, 'o', markersize=5, color='red')
            elif cell == n:
                plt.plot(x, y, 's', markersize=10, color='gray')

def label_pacman_on_image(bot_position, text_position, rot):
    """
    Labels Pacman on the current plot with its position and rotation.
    The position is given in grid coordinates, and the rotation is in degrees.
    """
    x, y = text_position
    plt.text(x, y, f"Pacman: ({bot_position[0]:.2f}, {bot_position[1]:.2f}, {np.rad2deg(rot):1.0f}°)", color='black', fontsize=12, va='top')

def calc_map_position_from_bounds(bot_loc, bounds):
    """
    Calculates the position of the bot in the map based on its 
    location and the bounds of the image.
    """

    botx, boty = bot_loc
    x_min, x_max, y_min, y_max = bounds
    gridw, gridh = len(grid[0]), len(grid)
    
    # Normalize the bot's position within the bounds
    norm_x = (botx - x_min) / (x_max - x_min)
    norm_y = (boty - y_min) / (y_max - y_min)
    
    # Scale to grid dimensions
    grid_x = norm_x * (gridw-1)
    grid_y = norm_y * (gridh-1)
    
    return grid_x, grid_y

def fig_to_rgb(fig):
    """
    Used to convert a matplotlib figure to an RGB image for video writing.
    """
    fig.canvas.draw()
    # RGBA buffer to RGB array
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

fig0 = plt.figure(10)
fig11 = plt.figure(11)
 
video_writer = None
output_path = "output_combined.mp4"
fps = 60

botx, boty, botrot = 0, 0, 0
pbar = tqdm(total=len(frames), desc="Processing frames", unit="frame")
for i, frame in enumerate(frames):
    # --- Process each frame ---
    plt.figure(10)
    plt.clf()
    frame = normalize_rotation_to_grid(frame)
    bounds = extract_bounds(frame, decimation = 2, ghost_home_direction=right, debug_display=False)
    gridw, gridh = len(grid[0]), len(grid)

    # find pacbot
    bot_loc = find_tag(frame)
    if bot_loc is not None:
        botx, boty, botrot = bot_loc

    # --- Display the results ---
    display_bounds(frame, bounds, fignum=10)
    draw_pacman_at(position=(botx, boty), rotation=botrot)
    label_pacman_on_image(text_position = (0,0), bot_position=(botx, boty), rot=botrot)

    plt.figure(11)
    plt.clf()

    # --- Draw the map based on the grid representation ---
    draw_map_from_grid()
    map_pos = calc_map_position_from_bounds(bot_loc=(botx, boty), bounds=bounds)
    draw_pacman_at(position=map_pos, direction_length=1,rotation=(np.round((botrot)/(np.pi/2))*(np.pi/2)))
    label_pacman_on_image(text_position = (0,-2), bot_position=map_pos, rot=np.round(botrot/90)*90)

    plt.pause(0.05)

    # --- Combine Figures into 1 Video Frame ---
    img0 = fig_to_rgb(fig0)
    img11 = fig_to_rgb(fig11)

    # Ensure heights match before horizontal stacking
    if img0.shape[0] != img11.shape[0]:
        img11 = cv2.resize(
            img11, (img11.shape[1], img0.shape[0]), interpolation=cv2.INTER_AREA
        )

    combined_frame = np.hstack((img0, img11))

    # Initialize VideoWriter on first iteration
    if video_writer is None:
        h, w, _ = combined_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    video_writer.write(combined_frame)
    pbar.update(1)
pbar.close()
    
if video_writer is not None:
    video_writer.release()
plt.close("all")