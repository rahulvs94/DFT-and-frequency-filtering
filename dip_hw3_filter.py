"""dip_hw3_dft.py: Starter file to run howework 2"""

#Example Usage: ./dip_hw3_filter -i image -m mask
#Example Usage: python dip_hw3_filter.py -i image -m mask


__author__      = "Pranav Mantini"
__email__ = "pmantini@uh.edu"
__version__ = "1.0.0"

import cv2
import sys
from DFT.Filtering import Filtering
from datetime import datetime


def display_image(window_name, image):
    """A function to display image"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     fitlering method and writes the output image"""

    # Parse input arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", "--image", dest="image",
                        help="specify the name of the image", metavar="IMAGE")
    parser.add_argument("-m", "--mask", dest="mask",
                        help="specify name of the mask (ideal_l, ideal_h, butterworth_l, butterworth_h, gaussian_l or gaussian_h)", metavar="MASK")
    parser.add_argument("-c", "--cutoff_f", dest="cutoff_f",
                        help="specify the cutoff frequency", metavar="CUTOFF FREQUENCY")
    parser.add_argument("-o", "--order", dest="order",
                        help="specify the order for butterworth filter", metavar="ORDER")

    args = parser.parse_args()

    # Load image
    if args.image is None:
        print("Please specify the name of image")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        image_name = args.image.split(".")[0]
        input_image = cv2.imread(args.image, 0)
        rows, cols = input_image.shape

    # Check resize scale parametes
    if args.mask is None:
        print("Mask not specified using default (ideal_l)")
        print("use the -h option to see usage information")
        mask = 'ideal_l'
    elif args.mask not in ['ideal_l', 'ideal_h', 'butterworth_l', 'butterworth_h', 'gaussian_l', 'gaussian_h']:
        print("Unknown mask, using default (ideal_l)")
        print("use the -h option to see usage information")
        mask = 'ideal_l'
    else:
        mask = args.mask

    if args.cutoff_f is None:
        print("Cutoff not specified using (min(height,widht)/2")
        print("use the -h option to see usage information")
        cutoff_f = min(rows, cols) / 2
    else:
        cutoff_f = float(args.cutoff_f)

    if mask in ['butterworth_l', 'butterworth_h']:
        if args.order is None:
            print("Order of the butterworth filter not specified, using default (2)")
            print("use the -h option to see usage information")
            order = 2
        else:
            order = float(args.order)
    output = None
    if mask in ['butterworth_l', 'butterworth_h']:
        Filter_obj = Filtering(input_image, mask, cutoff_f, order)
        output = Filter_obj.filtering()
    else:
        Filter_obj = Filtering(input_image, mask, cutoff_f)
        output = Filter_obj.filtering()

    # Write output file
    output_dir = 'output/'

    output_image_name = output_dir+image_name+"_"+mask+datetime.now().strftime("%m%d-%H%M%S")+".jpg"
    cv2.imwrite(output_image_name, output[0])
    output_image_name = output_dir + image_name+"_dft_" + mask + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_image_name, output[1])
    output_image_name = output_dir + image_name + "_dft_filter_" + mask + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_image_name, output[2])


if __name__ == "__main__":
    main()







