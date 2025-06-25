# Author: Kevyn Angueira Irizarry
# Created: 2025-03-06
# Last Modified: 2025-06-25

import cv2
import numpy as np
import os
import random
from pathlib import Path

def resize_for_display(image, max_width=500, max_height=400):
    """ Resize image while maintaining aspect ratio for display. """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def nothing(x):
    pass

def get_random_segment_images(base_dir, num_images=6):
    """ Randomly select segment images from various leaf/media paths. """
    base_dir = Path(base_dir)
    images = []
    filenames = []

    leaf_ids = [d for d in base_dir.iterdir() if d.is_dir()]
    if not leaf_ids:
        print("No leaf IDs found.")
        return [], []

    while len(images) < num_images:
        leaf_id = random.choice(leaf_ids)
        results_path = leaf_id / "defoliated" / "results"

        if not results_path.exists():
            continue

        media_ids = [d for d in results_path.iterdir() if d.is_dir()]
        if not media_ids:
            continue

        media_id = random.choice(media_ids)
        segments_path = media_id / "leafSegments"

        if not segments_path.exists():
            continue

        segment_images = list(segments_path.glob("*.jpg"))
        if not segment_images:
            continue

        chosen_img = random.choice(segment_images)
        img = cv2.imread(str(chosen_img))
        if img is not None:
            images.append(img)
            filenames.append(str(chosen_img))

    return images, filenames

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".jpg"):
            path = folder / filename
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)
    return images


# Load random segment images
#BASE_DIR = Path("LeafMedia")
#images, filenames = get_random_segment_images(BASE_DIR)
BASE_DIR = Path("/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/TestLab/Flatboard")
images = load_images_from_folder(BASE_DIR)

if not images:
    print("No segment images could be loaded.")
    exit()

# Convert all images to LAB
lab_images = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in images]

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for L, A, and B (lower and upper bounds)
cv2.createTrackbar("LL", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("LA", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("LB", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("UL", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UA", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UB", "Trackbars", 255, 255, nothing)

while True:
    # Get values from trackbars
    ll = cv2.getTrackbarPos("LL", "Trackbars")
    la = cv2.getTrackbarPos("LA", "Trackbars")
    lb = cv2.getTrackbarPos("LB", "Trackbars")
    ul = cv2.getTrackbarPos("UL", "Trackbars")
    ua = cv2.getTrackbarPos("UA", "Trackbars")
    ub = cv2.getTrackbarPos("UB", "Trackbars")

    lower_bound = np.array([ll, la, lb])
    upper_bound = np.array([ul, ua, ub])

    masks = [cv2.inRange(lab, lower_bound, upper_bound) for lab in lab_images]
    results = [cv2.bitwise_and(img, img, mask=mask) for img, mask in zip(images, masks)]

    # Stack each original + mask horizontally
    pairs = [
        np.hstack([resize_for_display(img), resize_for_display(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))])
        for img, mask in zip(results, masks)
    ]

    # Stack all pairs vertically
    final_display = np.vstack(pairs)

    # Show results
    cv2.imshow("Results", final_display)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
