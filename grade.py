import keras as ks
import os
import argparse
import cv2 as cv
import numpy as np
import csv
import tensorflow as tf

# parse arguments
parser = argparse.ArgumentParser(description='Grade students models based on their mean squared error and file size')

parser.add_argument('-p', '--path', help='path to models directory', required=True)
parser.add_argument('-o', '--output', help='csv file to output to', required=True)
parser.add_argument('-d', '--dataset', help='path to dataset', required=True)
parser.add_argument('-b', '--batch', help='Batch size', required=False, default=32)

args = parser.parse_args()

# global variables
model_attributes = []


def main():
    banner()
    models_path = format_paths(args.path)
    dataset_path = format_paths(args.dataset)
    output = args.output
    print_config(models_path, output, dataset_path)
    grade_models(models_path, dataset_path, args.batch)
    scores = calculate_score()
    for i in range(len(scores)):
        model_attributes[i].append(scores[i])
    
    output_csv(output)
    print(model_attributes)

def calculate_score():
    mse = []
    size = []
    for model in model_attributes:
        mse.append(model[2])
        size.append(model[1])
    print(mse)
    print(size)
    # normalize mse and size
    mse = [float(i)/sum(mse) for i in mse]
    size = [float(i)/sum(size) for i in size]

    # now we inverse the mses and sizes to get a score
    score = []
    for i in range(len(mse)):
        mse[i] = 1 - mse[i]
        size[i] = 1 - size[i]
        score.append(mse[i] * size[i])

    return score 
        

def output_csv(output):
    # if the csv file already exists, delete it
    if os.path.exists(output):
        os.remove(output)
    with open(output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Size', 'MSE', 'Score'])
        for row in model_attributes:
            writer.writerow(row)

def grade_models(models_path, dataset_path, batch_size):
    for root, dirs, files in os.walk(models_path):
        for file in files:
            attributes = []
            if file.endswith(".h5"):
                print("Grading model: {}".format(file))
                attributes.append(file.split(".h5")[0])
                attributes.append(get_size(os.path.join(root, file)))    
                attributes.append(compute_mse(os.path.join(root, file), dataset_path, batch_size))
            model_attributes.append(attributes)            

def compute_mse(model_path, dataset_path, batch_size):

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

    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='constant')

    def preprocess(img_path, steering):
        # decode the image and steering angle from tensor 
        img_path = img_path.numpy().decode('utf-8') # convert the tensor to a numpy object
        steering = steering.numpy() # convert the tensor to a numpy object
        img = cv.imread(img_path) # read the image

        augmented = augmentor.random_transform(img)
        # we convert the frame to the HSV color space
        hsv = cv.cvtColor(augmented, cv.COLOR_BGR2HSV)
        # blur the image to remove noise
        blur = cv.GaussianBlur(hsv, (5, 5), 0)
        # mask the image to get only the desired colors
        mask = cv.inRange(blur, (40, 10, 73), (110, 255, 240))
        # we erode and dilate to remove noise
        erode = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        dilate = cv.dilate(erode, np.ones((5, 5), np.uint8), iterations=1)
        # we smooth the image with some gaussian blur
        blur = cv.GaussianBlur(dilate, (5, 5), 0)
        resized = cv.resize(blur, (100, 66))
        flip = np.random.randint(0, 2)
        if flip == 1:
            resized = cv.flip(resized, 1)
            steering = 180 - steering
        resized = np.expand_dims(resized, axis=2) # add a dimension to the image
        return resized, steering

    def parse_function(img_path, steering):
        x_out, y_out = tf.py_function(preprocess, [img_path, steering], [tf.float32, tf.float32])
        # convert outputs to tensors

        x = tf.cast(x_out, tf.float32)
        y = tf.cast(y_out, tf.float32)
        return x, y

    # load the data
    image_paths, steering_angles = load_data(dataset_path)

    # create a dataset from the training data and make a data pipeline
    ds = tf.data.Dataset.from_tensor_slices((image_paths, steering_angles)) # create a dataset from the image paths and steering angles
    ds = ds.repeat() 
    ds = ds.map(parse_function , num_parallel_calls=tf.data.experimental.AUTOTUNE) # map the parse function to the dataset adding parallelism increases performance by over 8 times
    ds = ds.batch(batch_size) # batch the dataset
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE) # prefetch the dataset to improve performance
    
    model = ks.models.load_model(model_path)
    metrics = model.evaluate(ds, steps=len(image_paths) // batch_size) # evaluate the model on the dataset
    return metrics
    

def format_paths(path):
    path = path.replace("\\", "/")
    if path[-1] != "/":
        path += "/"
    return path


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

