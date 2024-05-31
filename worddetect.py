#step 1
# import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import streamlit as st

# sys.path.append(r'C:\Users\MOHD AREEF\RESEARCH&DEV\EMAGIA\OCR\handwriting-ocr\src')
  
# sys.path.append(r'C:\Users\MOHD AREEF\RESEARCH&DEV\EMAGIA\OCR\handwriting-ocr\data\pages\page09.jpg')
from ocr.helpers import implt, resize, ratio 
from ocr import page   

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (15.0, 10.0) 

# step2
st.title("Text Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = page.detection(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    st.write("uploaded image.")
    st.image(img, caption='Processed Image', use_column_width=True)


# image = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
# image = page.detection(image)
# img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# implt(img, 'gray')  

#step 3
    def sobel(channel):
        """ The Sobel Operator"""
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
        sobel = np.hypot(sobelX, sobelY)
        sobel[sobel > 255] = 255
        return np.uint8(sobel)

    def edge_detect(im):
        """ 
        Edge detection 
        The Sobel operator is applied for each image layer (RGB)
        """
        return np.max(np.array([sobel(im[:,:, 0]), sobel(im[:,:, 1]), sobel(im[:,:, 2]) ]), axis=0)

    # Image pre-processing - blur, edges, threshold, closing
    blurred = cv2.GaussianBlur(image, (5, 5), 18)
    edges = edge_detect(blurred)
    ret, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    bw_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))

    implt(edges, 'gray', 'Sobel operator')
    implt(bw_image, 'gray', 'Final closing')

# #STEP --------------------    4
# def union(a,b):
#     x = min(a[0], b[0])
#     y = min(a[1], b[1])
#     w = max(a[0]+a[2], b[0]+b[2]) - x
#     h = max(a[1]+a[3], b[1]+b[3]) - y
#     return [x, y, w, h]

# def intersect(a,b):
#     x = max(a[0], b[0])
#     y = max(a[1], b[1])
#     w = min(a[0]+a[2], b[0]+b[2]) - x
#     h = min(a[1]+a[3], b[1]+b[3]) - y
#     if w<0 or h<0:
#         return False
#     return True

# def group_rectangles(rec):
#     """
#     Uion intersecting rectangles
#     Args:
#         rec - list of rectangles in form [x, y, w, h]
#     Return:
#         list of grouped ractangles 
#     """
#     tested = [False for i in range(len(rec))]
#     final = []
#     i = 0
#     while i < len(rec):
#         if not tested[i]:
#             j = i+1
#             while j < len(rec):
#                 if not tested[j] and intersect(rec[i], rec[j]):
#                     rec[i] = union(rec[i], rec[j])
#                     tested[j] = True
#                     j = i
#                 j += 1
#             final += [rec[i]]
#         i += 1
            
#     return final 


# #STEP ------------------------------------ 5
# def textDetectWatershed(thresh, original):
#     """ Text detection using watershed algorithm """
#     # According to: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
#     img = resize(original, 3000)
#     thresh = resize(thresh, 3000)
#     # noise removal
#     kernel = np.ones((3,3),np.uint8)
#     opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    
#     # sure background area
#     sure_bg = cv2.dilate(opening,kernel,iterations=3)

#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#     ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg,sure_fg)
    
#     # Marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)

#     # Add one to all labels so that sure background is not 0, but 1
#     markers += 1

#     # Now, mark the region of unknown with zero
#     markers[unknown == 255] = 0
    
#     markers = cv2.watershed(img, markers)
#     implt(markers, t='Markers')
#     image = img.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Creating result array
#     boxes = []
#     for mark in np.unique(markers):
#         # mark == 0 --> background
#         if mark == 0:
#             continue

#         # Draw it on mask and detect biggest contour
#         mask = np.zeros(gray.shape, dtype="uint8")
#         mask[markers == mark] = 255

#         cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#         c = max(cnts, key=cv2.contourArea)
        
#         # Draw a bounding rectangle if it contains text
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
#         maskROI = mask[y:y+h, x:x+w]
#         # Ratio of white pixels to area of bounding rectangle
#         r = cv2.countNonZero(maskROI) / (w * h)
        
#         # Limits for text
#         if r > 0.1 and 2000 > w > 15 and 1500 > h > 15:
#             boxes += [[x, y, w, h]]
    
#     # Group intersecting rectangles
#     boxes = group_rectangles(boxes)
#     bounding_boxes = np.array([0,0,0,0])
#     for (x, y, w, h) in boxes:
#         cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 8)
#         bounding_boxes = np.vstack((bounding_boxes, np.array([x, y, x+w, y+h])))
        
#     implt(image)

#     # Recalculate coordinates to original size
#     boxes = bounding_boxes.dot(ratio(original, img.shape[0])).astype(np.int64)
#     return boxes[1:]  




#STEP ------------------------- 6

#MY FUNCTION
    import cv2
    import numpy as np

    def resize(image, width):
        """Resize image maintaining aspect ratio."""
        dim = None
        (h, w) = image.shape[:2]
        if w > width:
            r = width / float(w)
            dim = (width, int(h * r))
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            return resized
        return image

    def implt(img, t='Image'):
        """Display image."""
        from matplotlib import pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(t)
        plt.show()

    def group_rectangles(rectangles):
        """Group intersecting rectangles."""
        # Placeholder for grouping logic
        # In practice, you might use a clustering algorithm or another method
        # Here, we simply return the input list for simplicity
        return rectangles

    def ratio(image, target_height):
        """Calculate resize ratio."""
        return target_height / image.shape[0]

    def textDetectWatershed(thresh, original): 
        """ Text detection using watershed algorithm, targeting handwritten text """
        # Resize images
        img = resize(original, 3000)
        thresh = resize(thresh, 3000)
        
        # Noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=3)
        
        # Sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(img, markers)
        image = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        boxes = []
        for mark in np.unique(markers):
            if mark == 0:
                continue

            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == mark] = 255

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if not cnts:
                continue
            
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
            maskROI = mask[y:y+h, x:x+w]
            r = cv2.countNonZero(maskROI) / (w * h)
            
            if   w/h >1 :
                boxes += [[x, y, w, h]]
        
        boxes = group_rectangles(boxes)
        bounding_boxes = np.array([0,0,0,0])
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 8)
            text = f"w={w}, h={h}"
    #         cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            bounding_boxes = np.vstack((bounding_boxes, np.array([x, y, x+w, y+h])))
            
        # implt(image, 'Bounding rectangles')
        st.write("processed image")
        st.image(image, caption='Bounding rectangles', use_column_width=True)

        boxes = bounding_boxes.dot(ratio(original, img.shape[0])).astype(np.int64)
        return boxes[1:]   

    # Example usage:
    # thresh = cv2.imread('path_to_thresholded_image.png', cv2.IMREAD_GRAYSCALE)
    # original = cv2.imread('path_to_original_image.png')
    # boxes = textDetectWatershed(thresh, original)
    # print("Number of boxes:", len(boxes))
    boxes = textDetectWatershed(bw_image, image)
    print("Number of boxes:", len(boxes))


else:
    st.write("Please upload an image.")
