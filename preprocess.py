# Ex: python preprocess.py -csv oasis_cross-sectional.csv -p data/raw_train -mt train -combine_ad
import argparse
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import csv
from shutil import copyfile

csv_file = ""
data_input_path = ""
img_extensions = ['.jpg', '.jpeg', '.gif', '.png']
model_type = ""
combine_moderate_dementia = False # Because we don't have a lot of training data, combine 1.0 and 2.0 
                                  # CSR values for one moderate classification
combine_all_dementia = False # Combine CSR values for .5, 1.0, and 2.0 into one dementia classification

def build_unclassified_dementia_folder(individual_file_path, record, output_type):
    output_file_path = "data" + "/" + model_type + "/" + output_type + "/" + "unclassified_dementia"
    filename = individual_file_path.split('/')[len(individual_file_path.split('/'))-1]
    copyfile(individual_file_path, output_file_path + "/" + filename)
    print("Saved image " + filename + " " + "to unclassified_dementia folder in " + output_type + ".")
    return

def build_moderate_dementia_folder(individual_file_path, record, output_type):
    output_file_path = "data" + "/" + model_type + "/" + output_type + "/" + "moderate_dementia"
    filename = individual_file_path.split('/')[len(individual_file_path.split('/'))-1]
    copyfile(individual_file_path, output_file_path + "/" + filename)
    print("Saved image " + filename + " " + "to moderate_dementia folder in " + output_type + ".")
    return

def build_slight_dementia_folder(individual_file_path, record, output_type):
    output_file_path = "data" + "/" + model_type + "/" + output_type + "/" + "slight_dementia"
    filename = individual_file_path.split('/')[len(individual_file_path.split('/'))-1]
    copyfile(individual_file_path, output_file_path + "/" + filename)
    print("Saved image " + filename + " " + "to slight_dementia folder in " + output_type + ".")
    return

def build_mild_dementia_folder(individual_file_path, record, output_type):
    output_file_path = "data" + "/" + model_type + "/" + output_type + "/" + "mild_dementia"
    filename = individual_file_path.split('/')[len(individual_file_path.split('/'))-1]
    copyfile(individual_file_path, output_file_path + "/" + filename)
    print("Saved image " + filename + " " + "to mild_dementia folder in " + output_type + ".")
    return

def build_no_dementia_folder(individual_file_path, record, output_type):
    output_file_path = "data" + "/" + model_type + "/" + output_type + "/" + "no_dementia"
    filename = individual_file_path.split('/')[len(individual_file_path.split('/'))-1]
    copyfile(individual_file_path, output_file_path + "/" + filename)
    print("Saved image " + filename + " " + "to no_dementia folder in " + output_type + ".")
    return

def check_is_img(filename):
    is_img = False
    for ext in img_extensions:
        if ext in filename:
            is_img = True
    return is_img

def check_against_csv(each, subfolder_name, subfolder_path):
    # print(subfolder_path)
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for record in reader:
            # print(record['ID'], record['Age'], record['M/F'], record['CDR'])
            if record['ID'] == each:
                # print(record['ID'], subfolder_name, record['CDR'])
                for _each in os.listdir(subfolder_path):
                    # print(_each)
                    is_img = check_is_img(subfolder_path + "/" + _each)
                    if is_img:
                        # print(record['ID'], _each, record['CDR'])
                        output_type = subfolder_name
                        output_path = data_input_path + "/" + each + "/" + output_type
                        for individual_file in os.listdir(output_path):
                            is_img = check_is_img(individual_file)
                            if is_img:
                                individual_file_path = output_path + "/" + individual_file
                                # print(individual_file_path, record['CDR'])
                                if record['CDR'] is "0":
                                    build_no_dementia_folder(individual_file_path, record, output_type)
                                if record['CDR'] is "0.5":
                                    build_mild_dementia_folder(individual_file_path, record, output_type)
                                # Because we don't have a lot of training data, let's combine slight and moderate dementia levels
                                # into just the moderate classification.
                                if combine_moderate_dementia and combine_all_dementia:
                                    print("Too many parameters were given to preprocessing. Only one CDR \
                                           combination option can be used.")
                                    exit()
                                if combine_moderate_dementia and not combine_all_dementia:
                                    if (record['CDR'] is "2" or record['CDR'] is "1"):
                                        build_moderate_dementia_folder(individual_file_path, record, output_type)
                                if combine_all_dementia and not combine_moderate_dementia:
                                    if (record['CDR'] is "2" or record['CDR'] is "1" or record['CDR'] is ".5"):
                                        build_moderate_dementia_folder(individual_file_path, record, output_type)
                                if not combine_moderate_dementia and not combine_all_dementia:
                                    if record['CDR'] is "1":
                                        build_slight_dementia_folder(individual_file_path, record, output_type)
                                    if record['CDR'] is "2":
                                        build_moderate_dementia_folder(individual_file_path, record, output_type)
                                if record['CDR'] is "":
                                    build_unclassified_dementia_folder(individual_file_path, record, output_type)

def crossreference_files(data_input_path):
        # Go through every all the folders in the discs extracted from OASIS
        is_dir = os.path.isdir(data_input_path)
        if is_dir:
            all_folders = os.listdir(data_input_path)
            for each in all_folders:
                _is_dir = os.path.isdir(data_input_path + "/" + each)
                if _is_dir:
                    folders = os.listdir(data_input_path + "/" + each)
                    for subfolder in folders:
                        subfolder_path = data_input_path + "/" + each + "/" + subfolder
                        is_subfolder_dir = os.path.isdir(subfolder_path)
                        if is_subfolder_dir:
                            subfolder_name = subfolder
                            check_against_csv(each, subfolder_name, subfolder_path)
                   
if __name__ == '__main__':
    print("Now preparing data from " + data_input_path + " " + "for training models.")
    ap = argparse.ArgumentParser()
    ap.add_argument("-csv", "--csv", required=True,
        help="CSV file (script is currently formatted for cross-sectional records from Oasis)")
    ap.add_argument("-p", "--path", required=True,
        help="folder path containing input data for preprocessing")
    ap.add_argument("-mt", "--model_type", required=True,
        help="type of model to output preprocessed data (train, validation, or test)")
    ap.add_argument("-combine_md", "--combine_moderate_dementia", required=False,
        help="combines CDR of 1.0 and 2.0 into moderate classification", action='store_true')
    ap.add_argument("-combine_ad", "--combine_all_dementia", required=False,
        help="combines CDR of .5, 1.0, and 2.0 into moderate classification", action='store_true')
    args = vars(ap.parse_args())
    csv_file = args["csv"]
    data_input_path = args["path"]
    model_type = args["model_type"]
    combine_moderate_dementia = args["combine_moderate_dementia"]
    combine_all_dementia = args['combine_all_dementia']
    crossreference_files(data_input_path)