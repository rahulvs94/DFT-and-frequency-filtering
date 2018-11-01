# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import math
import numpy as np
import cv2

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order=0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.filter_name = filter_name
        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""

        d0 = cutoff
        rows, columns = shape
        mask = np.zeros((rows, columns), dtype=int)
        mid_R, mid_C = int(rows/2), int(columns/2)
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R)**2 + (j - mid_C)**2)
                if d <= d0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0

        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        # Hint: May be one can use the low pass filter function to get a high pass mask
        d0 = cutoff
        # rows, columns = shape
        # mask = np.zeros((rows, columns), dtype=int)
        mask = 1 - self.get_ideal_low_pass_filter(shape, d0)
        
        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        d0 = cutoff
        n = order
        rows, columns = shape
        mask = np.zeros((rows, columns))
        mid_R, mid_C = int(rows / 2), int(columns / 2)
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
                mask[i, j] = 1 / (1 + (d / d0) ** (2 * n))

        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        d0 = cutoff
        n = order
        rows, columns = shape
        mask = np.zeros((rows, columns))
        mid_R, mid_C = int(rows / 2), int(columns / 2)
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
                if d == 0:
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1 / (1 + (d0 / d) ** (2 * n))
        
        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        d0 = cutoff
        rows, columns = shape
        mask = np.zeros((rows, columns))
        mid_R, mid_C = int(rows / 2), int(columns / 2)
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
                mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0))

        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        d0 = cutoff
        # rows, columns = shape
        # mask = np.zeros((rows, columns), dtype=int)
        mask = 1 - self.get_gaussian_low_pass_filter(shape, d0)
        
        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        a = 0
        b = 255
        c = np.min(image)
        d = np.max(image)
        rows, columns = np.shape(image)
        image1 = np.zeros((rows, columns), dtype=int)
        for i in range(rows):
            for j in range(columns):
                if (d-c) == 0:
                    image1[i, j] = ((b - a) / 0.000001) * (image[i, j] - c) + a
                else:
                    image1[i, j] = ((b - a) / (d - c)) * (image[i, j] - c) + a

        return np.uint8(image1)


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv"""

        image = self.image
        cutoff = self.cutoff
        order = self.order
        filter_name = self.filter_name
        shape = np.shape(image)

        # Steps:
        # 1. Compute the fft of the image
        fft = np.fft.fft2(image)

        # 2. shift the fft to center the low frequencies
        shift_fft = np.fft.fftshift(fft)
        mag_dft = np.log(np.abs(shift_fft))
        dft = self.post_process_image(mag_dft)
        #cv2.imshow('shift_dft', dft)
        #cv2.waitKey(0)

        # 3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        if filter_name == 'butterworth_l' or filter_name == 'butterworth_h':
            mask = self.filter(shape, cutoff, order)
        else:
            mask = self.filter(shape, cutoff)
            #mask1 = self.post_process_image(mask)
            #cv2.imshow('mask', mask)
            #cv2.waitKey(0)

        # 4. filter the image frequency based on the mask (Convolution theorem)
        filtered_image = np.multiply(mask, shift_fft)
        mag_filtered_dft = np.log(np.abs(filtered_image)+1)
        filtered_dft = self.post_process_image(mag_filtered_dft)
        #cv2.imshow('filtered_dft', filtered_dft)
        #cv2.waitKey(0)

        # 5. compute the inverse shift
        shift_ifft = np.fft.ifftshift(filtered_image)

        # 6. compute the inverse fourier transform
        ifft = np.fft.ifft2(shift_ifft)

        # 7. compute the magnitude
        mag = np.abs(ifft)

        # 8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        # take negative of the image to be able to view it (use post_process_image to write this code)
        filtered_image = self.post_process_image(mag)
        #cv2.imshow('filtered_image', filtered_image)
        #cv2.waitKey(0)
        """Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """

        return [np.uint8(filtered_image), np.uint8(dft), np.uint8(filtered_dft)]
