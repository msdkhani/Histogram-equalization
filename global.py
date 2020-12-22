import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

def get_histogram(img):
    
    histogram = np.zeros(256)
    # loop through pixels and sum up counts of pixels
    for pixel in img:
        histogram[pixel] += 1
    # return our final result
    return histogram

if __name__ == "__main__":


    start = time.time()
    input_img = np.asarray(Image.open('images/test-img4.pgm'))
    width, height = input_img.shape[:2]

    # put pixels in a 1D array by flattening out img array
    flat = input_img.flatten()

    #getting histogram
    hist = get_histogram(flat)
    MN = width*height
    #finding the probability
    array_pdf = hist/MN

    #setting the curriculum density
    CDF = 0
    CDF_matrix = np.zeros(256)
    for i in range(1, 256):
        CDF = CDF + array_pdf[i]
        CDF_matrix[i] = CDF
    
    final_array = np.zeros(256)
    final_array = (CDF_matrix * 255)
    #finding nearest neighbor
    for i in range (1,256):
        final_array[i] = np.round(final_array[i])

    img_new = final_array[flat]

    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, input_img.shape)

    end = time.time()
    print(f"Total time taken: {np.round(end - start, decimals=3)} seconds")
    #saving image
    cv2.imwrite("test-img4.png", img_new)