# Author: Kevyn Angueira Irizarry
# Created: 2025-03-25
# Last Modified: 2025-03-25

import cv2
import numpy as np
import os

def resize_for_display(image, max_width=500, max_height=400):
    """ Resize image while maintaining aspect ratio for display. """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def nothing(x):
    pass

def load_images_from_folder(folder):
    """ Load all images from a given folder. """
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def get_subfolders(directory):
    """ Get a list of subfolders in the given directory. """
    return [name for name in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, name))]

# Load images from folder
base_folder = "/home/icicle/VSCode/LeafAnalysis/ColorTests/ColorImages" 
subfolders = get_subfolders(base_folder)

if not subfolders:
    print("No subfolders found in the directory.")
    exit()

print("Select a color by entering the corresponding number:")
for i, subfolder in enumerate(subfolders):
    print(f"{i}: {subfolder}")

while True:
    try:
        choice = int(input("Enter your choice: "))
        if 0 <= choice < len(subfolders):
            folder_path = os.path.join(base_folder, subfolders[choice])
            break
        else:
            print("Invalid selection. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Load images from the chosen folder
images, filenames = load_images_from_folder(folder_path)

if not images:
    print("No images found in the folder.")
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

    # Resize images for display
    resized_masks = [resize_for_display(mask) for mask in masks]
    resized_results = [resize_for_display(res) for res in results]

    # Stack images horizontally
    row2 = np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in resized_masks])
    row3 = np.hstack(resized_results)

    # Stack rows vertically
    final_display = np.vstack([row2, row3])

    # Show results
    cv2.imshow("Results", final_display)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
