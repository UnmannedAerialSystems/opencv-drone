import cv2
import numpy as np
import pytesseract


#TODO: try remove the background by looping through colors like red blue orange.


# Load the image
image = cv2.imread('realData/4.jpg') #WORKS WITH REAL DATA 4, 3, 1

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper thresholds for the blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create a mask for the blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow(" ROI", result)
cv2.waitKey(0)

alreadyChecked = []

# Find contours in the blue regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in sorted(contours, key=cv2.contourArea, reverse=True):

# Find the contour with the largest area

    largest_contour_first = contour
    area = cv2.contourArea(contour)
    if area < 1000: #THIS WILL NEED TO BE CONFIGURED BASED ON CAMERA
        continue

    if any(cv2.contourArea(c) == area for c in alreadyChecked):
        continue


    # Approximate the contour to get the shape
    epsilon = 0.02 * cv2.arcLength(largest_contour_first, True)
    approx = cv2.approxPolyDP(largest_contour_first, epsilon, True)

    # Determine the shape based on the number of vertices
    num_vertices = len(approx)
    print(num_vertices)
    if num_vertices == 3:
        shape = "Triangle"
    elif num_vertices == 4:
        shape = "Rectangle"
    elif num_vertices == 5:
        shape = "Pentagon"
    elif num_vertices == 6:
        shape = "Hexagon"
    elif num_vertices == 7:
        shape = "Heptagon"
    elif num_vertices == 8:
        shape = "Octagon"
    else:
        shape = "Circle"

    # Draw the contour on the result image
    cv2.drawContours(result, [largest_contour_first], 0, (0, 255, 0), 2)
    cv2.imshow(" ROI", result)
    cv2.waitKey(0)
    # Extract the region of interest (ROI) from the contour
    padding = 50  # Define the padding size
    x, y, w, h = cv2.boundingRect(largest_contour_first)

    # Add padding to the bounding rectangle
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(result.shape[1] - x, w + 2*padding)
    h = min(result.shape[0] - y, h + 2*padding)

    # Extract the ROI with padding
    roi = result[y:y+h, x:x+w]
    cv2.imshow("Regular ROI", roi)
    cv2.waitKey(0)

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 

    # Apply thresholding to the grayscale ROI
    _, threshold_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    final_roi = cv2.bitwise_not(threshold_roi)

    largest_contour = largest_contour_first - np.array([x, y])
    # set everything not within largest_contour to white
    for i in range(len(final_roi)):
        for j in range(len(final_roi[i])):
            if cv2.pointPolygonTest(largest_contour, (j, i), False) < 0:
                final_roi[i][j] = 255

    #create a mask for only the black pixels
    mask = cv2.inRange(final_roi, 0, 0)
    #find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour = contours[0]

    cv2.drawContours(final_roi, contours, 0, (0, 255, 0), 2)


    #eliminate everything outside of the contour
    for i in range(len(final_roi)):
        for j in range(len(final_roi[i])):
            if cv2.pointPolygonTest(largest_contour, (j, i), False) < 0:
                final_roi[i][j] = 255


    # Use pytesseract to extract the number from the thresholded ROI
    number = pytesseract.image_to_string(final_roi, config = '--psm 10')
    number = number.strip()
    #if the output isn't a letter or number, rotate the image 30 degrees and try again
    max_rotations = 20
    angle = 30
    while not number.isalnum() and not number.isalpha() and max_rotations > 0 or len(number) != 1:
        center = (final_roi.shape[1] // 2, final_roi.shape[0] // 2)

        # Generate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        final_roi = cv2.warpAffine(final_roi, rotation_matrix, (final_roi.shape[1], final_roi.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))        

        number = pytesseract.image_to_string(final_roi, config = '--psm 10')
        number = number.strip()
        max_rotations -= 1 
        print("Rotating image")
    

    # Output the shape and the number
    print("The shape of the contour is:", shape)
    print("The number inside the shape is:", number)
    # Display the image that pytesseract is reading
    cv2.imshow("Image passed to tesseract", final_roi)
    cv2.waitKey(0)
    alreadyChecked.append(largest_contour_first)

    cv2.destroyAllWindows()
