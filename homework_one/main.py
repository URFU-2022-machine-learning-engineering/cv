from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This function segments the fish from the image by creating a mask based on color ranges in HSV color space.

    Parameters:
    img (numpy.ndarray): The input image in BGR color space.

    Returns:
    numpy.ndarray: The segmented image after applying morphological operations.
    """

    # Convert the image from BGR to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color ranges for orange and white colors in HSV space
    light_orange = (1, 190, 150)
    dark_orange = (30, 255, 255)
    light_white = (60, 0, 200)
    dark_white = (145, 150, 255)

    # Create a mask by checking if the HSV image falls within the orange and white color ranges
    mask = cv2.inRange(img_hsv, light_orange, dark_orange) + cv2.inRange(img_hsv, light_white, dark_white)

    # Apply morphological operations (closing followed by opening) to remove small holes and noise in the mask
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()

    stage = "train" if args.is_train else "test"

    try:
        images = [img_path for img_path in Path("dataset", stage, "imgs").glob("*.jpg")]
    except FileNotFoundError:
        print("Image folder is empty!\nPlease download images into image folder")
        exit(1)

    masks = {img.name: segment_fish(cv2.imread(str(img))) for img in images}
    print(compute_ious(masks, str(Path("dataset", stage, "masks"))))
