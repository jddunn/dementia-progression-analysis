# Ex: python test_network.py -f data/test/FSL_SEG -t FSL_SEG
import os
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import random
import imutils

top_model_weights_path = ""
train_data_dir = "data/train"
validation_data_dir = "data/validation"

data_type = ""
img_width, img_height = 256, 256

def predict(image_path):
    print("Predicting " + image_path)
    filename = image_path.split('/')[len(image_path.split('/'))-1]
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('oasis_cross-sectional_class_indices' + '_' + data_type + '.npy').item()

    num_classes = len(class_dictionary)

    orig = cv2.imread(image_path)

    orig = imutils.resize(orig, width=600) # Make images bigger (training data is in high-quality format)

    image = load_img(image_path, target_size=(img_width, img_height))
    # image = cv2.resize(image, (img_width, img_height), interpolation = cv2.INTER_NEAREST)

    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    probs = model.predict_proba(bottleneck_prediction)
    classes = model.predict_classes(bottleneck_prediction)
    # print(str(classes))
    # print(str(probs))
    class_predicted = model.predict_classes(bottleneck_prediction)
    # print(str(class_predicted))
    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]
    # print()

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = str(inv_map[inID]) + " - " + str(probs)
    print(label)
    cv2.putText(orig, label, (20, 45),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    # cv2.imshow("Classification", orig)
    cv2.imwrite("test_results" + "/" + inv_map[inID] + "/" + filename, orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def send_from_dir(path):
    is_dir = os.path.isdir(path)
    if is_dir:
        for each in os.listdir(path):
            predict(path + "/" + each)
    else:
        predict(path)


if __name__ == '__main__':
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True,
        help="path to image file or directory of images to test")
    ap.add_argument("-t", "--type", required=True,
        help="type of dataset / model to train (options: FSL_SEG, PROCESSED, or RAW)")
    args = vars(ap.parse_args())
    data_type = args["type"]
    if data_type == 'FSL_SEG':
        img_width, img_height = 176, 208
    train_data_dir = train_data_dir + "/" + data_type
    validation_data_dir = validation_data_dir + "/" + data_type
    top_model_weights_path = "oasis_cross-sectional" + "_" + data_type + ".h5"
    path = args["file"]
    send_from_dir(path)
    cv2.destroyAllWindows()