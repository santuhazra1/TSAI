import numpy as np
import cv2
import math

# Function to pad white pixels
def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value 

# Function to get vertices of the rotated isosceles triangle
def getcarvertices(center_x, center_y, theta):

    # Create 4 vertices of a rectangle around center_x, center_y
    a = [center_x - 10, center_y - 5]
    b = [center_x - 10, center_y + 5]
    c = [center_x + 10, center_y - 5]
    d = [center_x + 10, center_y + 5]

    p, q = center_x, center_y

    # get new vertices of the rectangle after a rotation angle theta with the below mathematical formula
    a1 = (int(((a[0] - p) * math.cos(math.radians(theta))) - ((a[1] - q) * math.sin(math.radians(theta))) + p), int(((a[0] - p) * math.sin(math.radians(theta))) + ((a[1] - q) * math.cos(math.radians(theta))) + q ))
    b1 = (int(((b[0] - p) * math.cos(math.radians(theta))) - ((b[1] - q) * math.sin(math.radians(theta))) + p), int(((b[0] - p) * math.sin(math.radians(theta))) + ((b[1] - q) * math.cos(math.radians(theta))) + q ))
    c1 = (int(((c[0] - p) * math.cos(math.radians(theta))) - ((c[1] - q) * math.sin(math.radians(theta))) + p), int(((c[0] - p) * math.sin(math.radians(theta))) + ((c[1] - q) * math.cos(math.radians(theta))) + q ))
    d1 = (int(((d[0] - p) * math.cos(math.radians(theta))) - ((d[1] - q) * math.sin(math.radians(theta))) + p), int(((d[0] - p) * math.sin(math.radians(theta))) + ((d[1] - q) * math.cos(math.radians(theta))) + q ))
    
    # So, here a1 and b1 are two base pixel cordinates of isosceles triangle
    # To claculate the 3rd pixel cordinate get mid point of c1 and d1
    e = (int((c1[0] + d1[0])/2),int((c1[1] + d1[1])/2))

    return a1, b1, e

class CropImage(object):

    # Initialize Class with crop_size and scale_size
    def __init__(self, crop_size = 100, scale_size = 32, normalized = True):
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.normalized = normalized

    def crop(self, img, x, y, theta):

        # If image is normalized then convert normalized image pixels to 0-255 range
        if self.normalized == True:
            img = np.asarray(img) * 255
            img = np.clip(img, 0, 255).astype('uint8')

        # Pad Image with half of crop size to avoid error in cropping boundary pixels
        img = np.pad(img, self.crop_size // 2, pad_with, padder=255.)
        
        # Get new shifted centers after padding
        x = int(x) + self.crop_size // 2
        y = int(y) + self.crop_size // 2

        # Get the vertices of the triangle around the center x, y and of a rotation theta
        pt1,pt2,pt3 = getcarvertices(x,y,theta)
        vertices = np.array([pt1, pt2, pt3], np.int32)
        pts = vertices.reshape((-1, 1, 2))

        # Transpose the image before drawing triangle to align with cv2 co-ordinates
        img = cv2.transpose(img)

        # Draw lines withe vertices
        img = cv2.polylines(img, [pts], isClosed=True, color=(128), thickness=3)

        # Fill the inside of the Triangle for the lines drawn above
        img = cv2.fillPoly(img, [pts], color=(128))

        # Convert the image back as it was before triangle was drawn
        img = cv2.flip(img,flipCode=0)
        img=cv2.transpose(img)
        img=cv2.flip(img,flipCode=1)

        # Crop the Image of required size
        cropped_image = img[x - self.crop_size//2:x + self.crop_size//2, y - self.crop_size//2:y + self.crop_size//2]

        # resize the image of required size
        cropped_image = cv2.resize(cropped_image, dsize=(self.scale_size,self.scale_size), interpolation=cv2.INTER_CUBIC)

        # Increase the dimension of the to make it work for CNN
        if len(cropped_image.shape) == 2:
            cropped_image = np.expand_dims(cropped_image, axis=0)

        # normalize the image again
        if self.normalized == True:
            cropped_image = np.asarray(cropped_image)/255.0

        return cropped_image      

        

