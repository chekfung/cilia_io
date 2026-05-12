import nd2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import cv2
import argparse
import sys

def process_images(data_file, output_file_directory):
    try:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Input file not found: {data_file}")
        
        if not os.path.exists(output_file_directory):
            os.makedirs(output_file_directory)
            print(f"Created output directory: {output_file_directory}")
        
        data_file_name = os.path.splitext(os.path.basename(data_file))[0]
        full_filepath = os.path.join(output_file_directory, data_file_name)
        
        with nd2.ND2File(data_file) as f:
            z_stack = f.asarray()
            print(f"Image stack shape: {z_stack.shape}")
            
            total_images = z_stack.shape[0]
            for i in range(total_images):
                print(f"Processing Image: {i+1} / {total_images}")
                image = z_stack[i]

                # Normalize and preprocess
                image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                image_uint8 = image_normalized.astype(np.uint8)
                image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
                
                fix_filename = f"{full_filepath}_frame_{i}.png"
                cv2.imwrite(fix_filename, image_rgb)
                
                if not os.path.exists(fix_filename):
                    raise IOError(f"Failed to save image: {fix_filename}")
        
        print("Image processing completed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process ND2 file and save images.")
    parser.add_argument("data_file", help="Path to the input ND2 file")
    parser.add_argument("output_directory", help="Path to the output directory for saving images")
    
    args = parser.parse_args()
    
    process_images(args.data_file, args.output_directory)

if __name__ == "__main__":
    main()