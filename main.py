import cv2
import numpy as np

# Load the image
img = cv2.imread('binary_image.png')

# Resize the image for better performance (adjust scale_percent as needed)
scale_percent = 60  # you can adjust this based on your image size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blurring to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to highlight the grid and cells
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Use morphological closing to enhance grid lines and connect cells
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours of each cell in the grid
contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Define a minimum and maximum area threshold to filter out small contours (signatures)
min_contour_area = 70000  # Minimum area for grid cells (adjust as needed)
max_contour_area = 300000  # Maximum area for grid cells (adjust as needed)

# Define an aspect ratio range for grid cells
min_aspect_ratio = 0.5  # Minimum aspect ratio (to avoid detecting signatures)
max_aspect_ratio = 1000  # Maximum aspect ratio (adjust based on grid shape)

# Draw contours of each valid grid block
result = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    contour_area = w * h
    aspect_ratio = w / h
    
    # Filter out contours that are either too small or too large or have an irregular aspect ratio
    if min_contour_area < contour_area < max_contour_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the result with filtered grid blocks (ignoring signatures)
cv2.imshow('Filtered Grid Cells Only', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('filtered_grid_only2.jpg', result)
