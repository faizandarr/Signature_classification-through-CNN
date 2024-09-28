import cv2
import os
import numpy as np

# Define the path to the folder containing the images
input_folder = 'D:\\University\\Semester 7\\Generative AI\\Assignment_1\\Signature_classification\\Data'
output_folder = 'Contours'

# Create a main folder to store the cropped contours if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the highest existing folder index
existing_folders = [int(f) for f in os.listdir(output_folder) if f.isdigit()]
folder_idx = max(existing_folders, default=0) + 1  # Start from the next folder index

# Get a list of all image files in the input folder and sort them numerically
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))], 
                     key=lambda x: int(os.path.splitext(x)[0]))

# Process each image in the folder
for image_file in image_files:
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Image '{image_file}' not found or unable to load.")
        continue

    # Resize the image for better performance (adjust scale_percent as needed)
    scale_percent = 60  # you can adjust this based on your image size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to binary image using a global threshold
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Save the binary image for reference
    cv2.imwrite(os.path.join(output_folder, f'binary_{image_file}'), binary_image)

    # Apply Gaussian blurring to reduce noise
    blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Apply adaptive thresholding to highlight the grid and cells
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Use morphological closing to enhance grid lines and connect cells
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of each cell in the grid
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define a minimum and maximum area threshold to filter out small contours (signatures)
    min_contour_area = 195000  # Minimum area for grid cells (adjust as needed)
    max_contour_area = 300000  # Maximum area for grid cells (adjust as needed)

    # Define an aspect ratio range for grid cells
    min_aspect_ratio = 0.5  # Minimum aspect ratio (to avoid detecting signatures)
    max_aspect_ratio = 1000  # Maximum aspect ratio (adjust based on grid shape)

    # Filter and collect valid contours
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_area = w * h
        aspect_ratio = w / h
        
        # Filter out contours that are either too small or too large or have an irregular aspect ratio
        if min_contour_area < contour_area < max_contour_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            valid_contours.append((x, y, w, h))

    # Sort the contours first by y-coordinate (top to bottom), then by x-coordinate (left to right)
    valid_contours = sorted(valid_contours, key=lambda b: (b[1], b[0]))

    # Draw contours and save them in the sorted order
    result = img.copy()
    contour_idx = 0  # Count of contours saved in the current folder
    contour_count = 0  # Count of contours saved in the current folder

    for (x, y, w, h) in valid_contours:
        # Create a subfolder for the current serial number if it doesn't exist
        subfolder_path = os.path.join(output_folder, str(folder_idx))
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Draw a rectangle around the contour
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the contour and save it
        cropped_contour = img[y:y + h, x:x + w]
        contour_filename = os.path.join(subfolder_path, f'contour_{contour_idx}.jpg')
        cv2.imwrite(contour_filename, cropped_contour)

        contour_idx += 1
        contour_count += 1

        # After every 4 images, increment the folder index
        if contour_count % 4 == 0:
            folder_idx += 1

    # Show the result with filtered grid blocks (ignoring signatures)
    #cv2.imshow(f'Filtered Grid Cells Only - {image_file}', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the result for the current image
    cv2.imwrite(os.path.join(output_folder, f'Whole_{image_file}'), result)
