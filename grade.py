import keras as ks
import os
import argparse
import cv2 as cv
import numpy as np
import csv

# parse arguments
parser = argparse.ArgumentParser(description='Grade students models based on their mean squared error and file size')

parser.add_argument('-p', '--path', help='path to models directory', required=True)
parser.add_argument('-o', '--output', help='path to output file', required=True)
parser.add_argument('-d', '--dataset', help='path to dataset', required=True)

args = parser.parse_args()

# global variables
model_attributes = []


def main():
    banner()
    print_config(args.path, args.output, args.dataset)
    grade_models(args.path, args.output, args.dataset)
    print(model_attributes)

def grade_models(models_path, output_path, dataset_path):

    for root, dirs, files in os.walk(models_path):
        for file in files:
            attributes = []
            if file.endswith(".h5"):
                print("Grading model: {}".format(file))
                attributes.append(file)
                attributes.append(get_size(os.path.join(root, file)))      
            model_attributes.append(attributes)            

def compute_mse(model_path, dataset_path):

    def load_data(directory):
        image__paths = []
        csv_file = ""
        for file in os.listdir(directory): # for each file in the directory
            if file.endswith(".jpg"): # if the file is an image
                image__paths.append(directory + file) # add the image path to the list
            if file.endswith(".csv"): # if the file is a csv file
                csv_file = file # we save it for later

        # now our files are in the train list we need to sort them from smallest file name to largest. The file name is the exact time the image was taken.
        image__paths.sort(key=lambda x: int(x.split('/')[-1][:-4])) # the lambda function returns the numbers in the file name

        # now we need to read the csv file and get the steering angles
        with open(directory + csv_file, 'r') as f:
            reader = csv.reader(f) # create a reader object
            steering_angles = [] # create a list to store the steering angles
            for row in reader: # for each row in the csv file
                steering_angles.append(float(row[0])) # add the steering angle to the list
        return image__paths, steering_angles # return the image paths and steering angles

    def batch_generator(image_paths, steering_angles, batch_size):
        while True:
            for start in range(0, len(image_paths), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(image_paths))
                ids_batch = image_paths[start:end]
                for id in ids_batch:
                    img = cv.imread(id)
                    img = cv.resize(img, (100, 66))
                    # we convert the frame to the HSV color space
                    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                    # blur the image to remove noise
                    blur = cv.GaussianBlur(hsv, (5, 5), 0)
                    # mask the image to get only the desired colors
                    mask = cv.inRange(blur, (40, 25, 73), (93, 194, 245))
                    # we erode and dilate to remove noise
                    erode = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
                    dilate = cv.dilate(erode, np.ones((5, 5), np.uint8), iterations=1)
                    # we smooth the image with some gaussian blur
                    blur = cv.GaussianBlur(dilate, (5, 5), 0)
                    x_batch.append(blur)
                    y_batch.append(steering_angles[image_paths.index(id)])
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.float32)
                yield x_batch, y_batch

    # load the data
    image_paths, steering_angles = load_data(dataset_path)

    batch_size = 32
    test_gen = batch_generator(image_paths, steering_angles, batch_size)
    
    model = ks.models.load_model(model_path)
    model.predict(test_gen, batch_size=batch_size)



def get_size(path):
    return os.path.getsize(path)

def print_config(models_path, output_path, dataset_path):
    model_count = 0
    for root, dirs, files in os.walk(models_path):
        for file in files:
            if file.endswith(".h5"):
                model_count += 1
    
    image_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                image_count += 1
    
    print("Models path: {}".format(models_path))
    print("Dataset path: {}".format(dataset_path))
    print("Output path: {}".format(output_path))
    print("Number of models: {}".format(model_count))
    print("Number of dataset images: {}".format(image_count))
    print("""
---------------------------------------------------------------------------------------------------------------------------
    """)





def banner():
    print("""
.        :       ...    :::::::-.  .,::::::   :::           .,-:::::/ :::::::..    :::.   :::::::-.  .,:::::: :::::::..   
;;,.    ;;;   .;;;;;;;.  ;;,   `';,;;;;''''   ;;;         ,;;-'````'  ;;;;``;;;;   ;;`;;   ;;,   `';,;;;;'''' ;;;;``;;;;  
[[[[, ,[[[[, ,[[     \[[,`[[     [[ [[cccc    [[[         [[[   [[[[[[/[[[,/[[['  ,[[ '[[, `[[     [[ [[cccc   [[[,/[[['  
$$$$$$$$"$$$ $$$,     $$$ $$,    $$ $$\"\"\"\"    $$'         "$$c.    "$$ $$$$$$c   c$$$cc$$$c $$,    $$ $$\"\"\"\"   $$$$$$c    
888 Y88" 888o"888,_ _,88P 888_,o8P' 888oo,__ o88oo,.__     `Y8bo,,,o88o888b "88bo,888   888,888_,o8P' 888oo,__ 888b "88bo,
MMM  M'  "MMM  "YMMMMMP"  MMMMP"`   \"\"\"\"YUMMM\"\"\"\"YUMMM       `'YMUP"YMMMMMM   "W" YMM   ""` MMMMP"`   \"\"\"\"YUMMMMMMM   "W" 
---------------------------------------------------------------------------------------------------------------------------
    """)








if __name__ == "__main__":
    main()

