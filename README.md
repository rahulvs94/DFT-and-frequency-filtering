# Digital Image Processing 

1. DFT:

	Code for computing forward fourier transform, inverse fourier transform, discrete cosine transfrom and magnitude of the fourier transform. 
	Input is a 2D matrix of size 15X15.

  - DFT/DFT.py file has functions "forward_transform", "inverse_transform", "discrete_cosine_tranform" and "magnitude"
  - I have implemented own code for all the computations, did not use inbuilt functions like "fft" or "dft" from numpy, opencv or other libraries
  - Usage: 
        python dip_hw3_dft.py
		
  - There is no output image or file for this part. Everything gets printed.
  

2. Frequency Filtering:

	Code to perfrom image filtering in the frequency domain by modifying the DFT of images using different masks. 
	Filter images using six different filters: ideal low pass (ideal_l), ideal high pass (ideal_h), butterworth low pass (butterworth_l), 
	butterworth high pass (butterworth_h), gaussian low pass (gaussian_l) and gaussian high pass filter (gaussian_h). 
	The input to your program is an image, name of the mask, cuttoff frequency and order (only valid for butterworth filter).

	- DFT/Filtering.py:
		 - \__init__(): Will intialize the required variable for filtering (image, mask function, cutoff, order). There is no need to edit this function  
		 - get_mask_freq_pass_filter(): There are six function definitions one for each of the filter. I haev written code to generate the masks for each filter here.
		 - filtering(): 
			- Code to perform image filtering here. The steps have been provided as a guideline for filtering. 
			- All the variable have already been intialized and can be used as self.image, self.cutoff, etc. 
			- The varaible self.filter is a handle to each of the six fitler functions. You can call it using self.filter(shape, cutoff, ...)
			- The function returns three images, filtered image, magnitude of the DFT and magnitude of filtered dft 
			- To be able to display magnitude of the DFT and magnitude of filtered dft, perform a logrithmic compression and convert the value to uint8
		 - post_process_image(): After fitlering and computing the inverse DFT, scale the image pixels to view it. I have written code to do a full contrast stretch here. 
	- In this part of the assignment, for computing fft, ifft and shifted fft, I have used inbuilt functions.
	- Usage: 
			python dip_hw3_filter.py -i [image] -m [ideal_l]-c [50]
			
			Below command is only for butterworth filters: 
			python dip_hw3_filter.py -i [image] -m [butterworth_l]-c [50] -o [2]
	  
  - Any output images or files must be saved to "output/" folder (dip_hw3_filter.py automatically does this)