import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time


def get_histogram(img):
    histogram = np.zeros(256)
    # loop through pixels and sum up counts of pixels
    for pixel in img:
        histogram[int(pixel)] += 1
    # return our final result
    return histogram


# the same as global histogram
def histogram_equalize(img):

    width, height = img.shape[:2]
    
    # put pixels in a 1D array by flattening out img array
    flat = img.flatten()

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
    img_new = np.reshape(img_new, img.shape)

    return img_new


if __name__ == "__main__":

    input_img = np.asarray(Image.open('images/test-img7.pgm'))
    width, height = input_img.shape[:2]

    #setting the windows size 
    window_size = 21
    window = np.zeros((window_size,window_size),dtype='uint8')

    #middle point in the windows (M-1/2)
    mid_point = window_size//2

    #zero pad the image from all side
    img_pad = np.pad(input_img, pad_width=((mid_point, mid_point),(mid_point, mid_point)), mode='constant', constant_values=0)

    new_img = np.zeros([input_img.shape[0], input_img.shape[1]],dtype='uint8')
        
    new_window = np.zeros((window_size,window_size))
    start = time.time()
    for i in range(0,input_img.shape[0]):
        for j in range(0,input_img.shape[1]):
            window = np.array(img_pad[i:i+window_size , j:j+window_size ])
            
            #calculate the histogram of the window
            new_window = histogram_equalize(window)
            
            #mapping the center pixel to new image(output)
            new_img[i,j]=new_window[mid_point,mid_point]

    end = time.time()
    print(f"Total time taken: {np.round(end - start, decimals=3)} seconds")
    #save the image
    cv2.imwrite("test-img7.png", new_img)
