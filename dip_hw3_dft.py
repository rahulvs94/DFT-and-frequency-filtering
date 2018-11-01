import cv2
import sys
from DFT import DFT
from numpy.random import rand
import numpy as np
from datetime import datetime

"""dip_hw3_dft.py: Starter file to run howework 3"""

#Example Usage: ./dip_hw3_dft
#Example Usage: python dip_hw3_dft


__author__      = "Pranav Mantini"
__email__ = "pmantini@uh.edu"
__version__ = "1.0.0"


def display_image(window_name, image):
    """A function to display image"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     interpolation method and writes the output image"""

    # intialize a matrix of 25X25 pixels
    input_matrix = np.int_(rand(15,15)*256)
    input_matrix = np.int_(rand(15,15)*256)
    print("---------------Input Matrix----------------")
    print(input_matrix)

    dft_obj = DFT.DFT()

    # Compute DFT
    fft_matrix = dft_obj.forward_transform(input_matrix)
    print("---------------Forward Fourier Transform----------------")
    print(fft_matrix)

    # Compute the inverse Fourier transform
    ift_matrix = dft_obj.inverse_transform(fft_matrix)
    print("---------------Inverse Fourier Transform----------------")
    print(ift_matrix)
    
    # Compute the magnitude of the dft
    magnitude_matrix = dft_obj.magnitude(ift_matrix)
    print("---------------Magnitude of the inverse Forward Fourier Transform ----------------")
    print(magnitude_matrix)

    # Compute the discrete cosine transform
    dct_matrix = dft_obj.discrete_cosine_tranform(input_matrix)
    print("---------------Discrete Cosine Transform----------------")
    print(dct_matrix)



if __name__ == "__main__":
    main()







