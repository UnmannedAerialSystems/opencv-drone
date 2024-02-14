import cv2
import numpy as np
import pytesseract


#TODO: try remove the background by looping through colors like red blue orange.


# Load the image
image = cv2.imread('realData/3.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper thresholds for the grass color in HSV
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# Create a mask for the grass color
mask = cv2.inRange(hsv, lower_green, upper_green)

# Invert the mask to get the non-grass regions
non_grass = cv2.bitwise_not(mask)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=non_grass)

# Find contours in the non-grass regions
contours, _ = cv2.findContours(non_grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour to get the shape
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Determine the shape based on the number of vertices
num_vertices = len(approx)
print(num_vertices)
if num_vertices == 3:
    shape = "Triangle"
elif num_vertices == 4:
    shape = "Rectangle"
elif num_vertices == 5:
    shape = "Pentagon"
else:
    shape = "Circle"

# Draw the contour on the result image
cv2.drawContours(result, [largest_contour], 0, (0, 255, 0), 2)
cv2.imshow(" ROI", result)
cv2.waitKey(0)
# Extract the region of interest (ROI) from the contour
padding = 50  # Define the padding size
x, y, w, h = cv2.boundingRect(largest_contour)

# Add padding to the bounding rectangle
x = max(0, x - padding)
y = max(0, y - padding)
w = min(result.shape[1] - x, w + 2*padding)
h = min(result.shape[0] - y, h + 2*padding)

# Extract the ROI with padding
roi = result[y:y+h, x:x+w]

# Convert the ROI to grayscale
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


# Apply thresholding to the grayscale ROI
_, threshold_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# isolate the number from the ROI
final_roi = cv2.bitwise_and(roi, roi, mask=threshold_roi)

#now set everything not in the threshold to white and everything in the threshold to black
#this is to make it easier for pytesseract to read
#also set anything outside of the shape to white
for i in range(len(final_roi)):
    for j in range(len(final_roi[i])):
        if final_roi[i][j][0] == 0 and final_roi[i][j][1] == 0 and final_roi[i][j][2] == 0:
            final_roi[i][j] = [255, 255, 255]
        else:
            final_roi[i][j] = [0, 0, 0]

#find the outline of the number from the final_roi
#convert to grayscale
gray_final_roi = cv2.cvtColor(final_roi, cv2.COLOR_BGR2GRAY)
#apply thresholding
_, threshold_final_roi = cv2.threshold(gray_final_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#find contours
contours, _ = cv2.findContours(threshold_final_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort the contours by area
contours = sorted(contours, key=cv2.contourArea, reverse=True)
#the largest contour is the number
largest_contour = contours[0]

cv2.drawContours(final_roi, contours, 0, (0, 255, 0), 2)

#eliminate everything outside of the contour
for i in range(len(final_roi)):
    for j in range(len(final_roi[i])):
        if cv2.pointPolygonTest(largest_contour, (j, i), False) < 0:
            final_roi[i][j] = [255, 255, 255]


# Use pytesseract to extract the number from the thresholded ROI
number = pytesseract.image_to_string(final_roi, config = '--psm 10 digits')

# Output the shape and the number
print("The shape of the contour is:", shape)
print("The number inside the shape is:", number)
# Display the image that pytesseract is reading
cv2.imshow("Thresholded ROI", final_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
