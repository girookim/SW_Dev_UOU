# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:35:23 2024

@author: UOU
"""

import cv2, os
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

folder_path = r"C:\\Users\\girookim\\Desktop\\SW_Dev\\Local_repo\\pre_dataset"
file_names = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]


results = model(folder_path+'\\'+file_names[0])

# Load the image using cv2
image = cv2.imread(folder_path+'\\'+file_names[0])

# Parameters for drawing
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates of the region

# List to store segmentation points
annotations = []

# Mouse callback function to draw contours
def draw_contour(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        annotations.append([(x, y)])  # Start a new contour

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Add points to the current contour
            annotations[-1].append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Close the contour by connecting the last point to the first
        annotations[-1].append((x, y))

# Function to display the image and collect annotations
def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return

    # Create a clone of the image for annotation display
    annotated_image = image.copy()
    cv2.namedWindow("Image Segmentation")
    cv2.setMouseCallback("Image Segmentation", draw_contour)

    while True:
        # Show the annotations on the cloned image
        temp_image = annotated_image.copy()
        for contour in annotations:
            points = np.array(contour, dtype=np.int32)
            cv2.polylines(temp_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the image with annotations
        cv2.imshow("Image Segmentation", temp_image)
        
        # Press 's' to save annotations, 'c' to clear, and 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save annotations
            with open("annotations.txt", "w") as f:
                for contour in annotations:
                    f.write(str(contour) + "\n")
            print("Annotations saved to annotations.txt")
        elif key == ord("c"):
            # Clear annotations
            annotations.clear()
            annotated_image = image.copy()
            print("Annotations cleared")
        elif key == ord("q"):
            break
        



# Draw bounding boxes and labels on the image
for result in results:  # Iterate through results (one per detection)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        label = box.cls[0]  # Class label index
        label_text = f"{model.names[int(label)]} {confidence:.2f}"
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the label
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('YOLO Detections', image)
cv2.waitKey(0)

        
        
        

cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    PathNames = r"C:\\Users\\girookim\\Desktop\\SW_Dev\\Local_repo\\pre_dataset"
    segment_image(PathNames + "//000000000872.jpg")
