import cv2
import numpy as np

def crop_leaf(frame, threshold=10):
    """
    Crop out the black borders around the leaf using thresholding and contour detection.
    Assumes that the leaf is the largest bright region in the frame.
    Returns the cropped leaf image and its bounding box.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return frame[y:y+h, x:x+w], (x, y, w, h)

def pad_image_to_match(image, target_shape):
    """
    Pads an image with zeroes to match the target shape.
    Returns the padded image and the y offset used in padding.
    """
    h, w = image.shape
    target_h, target_w = target_shape
    
    # Ensure the padded image is at least as high as 20
    pad_h = max(30, h, target_h)
    pad_w = max(target_w, w)
    
    padded = np.zeros((pad_h, pad_w), dtype=np.uint8)
    start_y = (pad_h - h) // 2
    start_x = (pad_w - w) // 2
    padded[start_y:start_y + h, start_x:start_x + w] = image
    
    return padded, start_y

def extract_bottom_band(image, band_height=20):
    """
    Extract a horizontal band (the lowest band) from the image.
    """
    return image[-band_height:, :]

def stitch_images(image1, image2, last_size):
    """ 
    Stitch two images vertically using template matching, but apply the
    stitching based on the original images (image1 and image2).
    
    image1: the top segment (original image)
    image2: the bottom segment (original image)
    
    The function first crops each image to get the leaf region, then uses the
    bottom 20-pixel band of image1's cropped leaf as a template to find the best
    overlap in image2 (after padding its cropped region to match image1's size).
    """

    template = image1[last_size:, :]
    
    # Crop the leaf regions and get bounding boxes
    cropped1, bbox1 = crop_leaf(template)
    cropped2, bbox2 = crop_leaf(image2)
    
    # Convert to grayscale for template matching
    gray1 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY)
    
    # Pad gray2 to match gray1's shape
    padded_gray2, pad_offset = pad_image_to_match(gray2, gray1.shape)
    
    # Extract the bottom band of gray1 as the template
    band_height = 30
    template_band = extract_bottom_band(gray1, band_height)
    
    # Template matching
    result = cv2.matchTemplate(padded_gray2, template_band, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"Template matching score: {max_val:.2f}") 

    if max_val > 0.2:

        # Determine where to stitch: use the original coordinates from the cropping
        stitch_y = last_size + bbox1[3]
        print(f"Stitch Y: {stitch_y}")

        # Determine overlap (adjust using the padding offset)
        overlap_height = max_loc[1] - pad_offset # Stitch location - top padding 
        non_overlapping_height = image2.shape[0] - (bbox2[1] + overlap_height) # img2 height - (top cord of cropped + overlap area + bottom padding)
        print(f"Overlap Height: {overlap_height}")
        print(f"Non-Overlap Height: {non_overlapping_height}")
        print(f"Pad Offset: {pad_offset}")
    

        # The new image height is the height of image1 up to its bottom (stitch_y) plus the non-overlapping\n
        # portion of image2. We use bbox2 and overlap_height from the second image.\n

        new_height =  max(stitch_y + non_overlapping_height, image1.shape[0])
        #new_height = stitch_y + (image2.shape[0] - (bbox2[1] + overlap_height))
        stitched_width = max(image1.shape[1], image2.shape[1])
        
        # Create the stitched image canvas
        stitched_image = np.zeros((new_height, stitched_width, 3), dtype=np.uint8)
        # Place image1 (top segment)
        print(f"image1 shape: {image1.shape}")
        print(f"stitched_image shape: {stitched_image.shape}")

        stitched_image[:stitch_y - overlap_height, :image1.shape[1]] = image1[:stitch_y - overlap_height, :]
        # Place image2 (bottom segment) starting at the determined stitch_y\n
        stitched_image[stitch_y - overlap_height:new_height, :image2.shape[1]] = image2[pad_offset:overlap_height+non_overlapping_height+pad_offset, :]
        #stitched_image[stitch_y:, :image2.shape[1]] = image2[bbox2[1] + overlap_height:, :]
        
        # Draw a horizontal red line to indicate the stitching boundary
        #cv2.line(stitched_image, (0, stitch_y), (stitched_width, stitch_y), (0, 0, 255), 2)

        cv2.imshow("Stitched Video Result", stitched_image)
        cv2.moveWindow("Stitched Video Result", 0, 200)
        cv2.imshow("Cropped 1", cropped1)
        cv2.moveWindow("Cropped 1", 1000, 200)
        cv2.imshow("Template Band", template_band)
        cv2.moveWindow("Template Band", 1000, 500)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite("LeafSegments/stitched_leaf_video.png", stitched_image)
    
        return stitched_image, stitch_y-overlap_height
    else: 
        return image1, image1.shape[0]

def process_video(video_path, frame_interval=10, video_duration=10):
    """
    Process a video of duration video_duration seconds.
    Extract a frame every frame_interval frames, resize each to 650x100,
    and stitch them sequentially.
    
    Returns the final stitched image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(min(total_frames, fps * video_duration))
    
    stitched_result = None
    frame_count = 0
    last_size = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every frame_interval-th frame
        if frame_count % frame_interval == 0:
            # Resize frame to 650x100 if not already that size
            frame_resized = cv2.resize(frame, (650, 100))
            print(f"Processing frame {frame_count}")
            if stitched_result is None:
                stitched_result = frame_resized
            else:
                # Stitch the current stitched result with the new frame
                stitched_result, last_size = stitch_images(stitched_result, frame_resized, last_size)
        
        frame_count += 1
    
    cap.release()
    return stitched_result

# Example usage:
video_path = "TestImages/leaf_output.mp4"  # Replace with your video file path
final_stitched_image = process_video(video_path, frame_interval=10, video_duration=10)

if final_stitched_image is not None:
    cv2.imshow("Stitched Video Result", final_stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("LeafSegments/stitched_leaf_video.png", final_stitched_image)
else:
    print("No frames were processed.")
