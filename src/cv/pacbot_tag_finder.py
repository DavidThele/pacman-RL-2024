import cv2
from pupil_apriltags import Detector
import numpy as np

def find_tag(image, family="tag36h11"):
    """
    Finds the position and orientation of the Pacman tag in the given image.
    Returns the center (cx, cy) and orientation (angle in radians) of the tag.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = Detector(
        families="tag36h11"
    )
    detections = detector.detect(gray)
    for tag in detections:
        # Center (cx, cy) in image pixels
        cx, cy = tag.center
        
        # Corners are ordered: [bottom-left, bottom-right, top-right, top-left]
        # Vector along bottom edge (corners[0] -> corners[1]) defines local +X axis
        p0 = tag.corners[0]
        p1 = tag.corners[1]
        
        # 2D angle in degrees
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle_rad = np.arctan2(dy, dx)
        return cx, cy, angle_rad