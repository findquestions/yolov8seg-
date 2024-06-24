import copy
import cv2
import os
import shutil
import numpy as np

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images_in_directory(directory, label_save_path):
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.png'):
            name = file.split('.')[0]
            file_path = os.path.join(directory, name + '.png')
            img = cv2.imread(file_path)
            H, W = img.shape[0:2]
            print(H, W)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnt, hit = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

            cnt = list(cnt)
            label_file = os.path.join(label_save_path, f"{name}.txt")
            with open(label_file, "a+") as f:
                for j in cnt:
                    result = []
                    pre = j[0]
                    for i in j:
                        if abs(i[0][0] - pre[0][0]) > 1 or abs(i[0][1] - pre[0][1]) > 1:  # Adjust interval points if needed
                            pre = i
                            temp = list(i[0])
                            temp[0] /= W
                            temp[1] /= H
                            result.append(temp)

                    if len(result) != 0:
                        f.write("0 ")
                        for line in result:
                            line = str(line)[1:-2].replace(",", "")
                            f.write(line + " ")
                        f.write("\n")

# Define the directories
base_directory = "D:\\Download\\yolov8\\NEU_Seg-main\\annotations"
test_directory = os.path.join(base_directory, "test")
training_directory = os.path.join(base_directory, "training")
label_base_directory = "D:\\Download\\yolov8\\NEU_Seg-main\\labels"

# Ensure the base label directory exists
ensure_directory_exists(label_base_directory)

# Define the label subdirectories for test and training
test_label_directory = os.path.join(label_base_directory, "test")
training_label_directory = os.path.join(label_base_directory, "training")

# Ensure the label subdirectories exist
ensure_directory_exists(test_label_directory)
ensure_directory_exists(training_label_directory)

# Process images in both directories
process_images_in_directory(test_directory, test_label_directory)
process_images_in_directory(training_directory, training_label_directory)
