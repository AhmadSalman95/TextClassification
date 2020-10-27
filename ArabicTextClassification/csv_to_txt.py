import csv
import pandas as pd
import numpy as np
import os
from testdata.prprocessArabicText import preprocess_text


def csv_to_txt_train_test_group_without_preprocess(in_file_path: str, name_of_class: str, columns: list):
    """this function to split the csv to 80% train,20% test then convert every line to [index].txt
    parameters:
    inFilePath:string path of csv file
    nameOfClass:string name of classes
    columns:list string what column name you need to detect """
    # split the data to train 80% and test 20%
    in_file = pd.read_csv(in_file_path)
    in_file['split'] = np.random.randn(in_file.shape[0], 1)
    msk = np.random.rand(len(in_file)) <= 0.8
    train_file = in_file[msk]
    test_file = in_file[~msk]
    in_file_pathAbs = os.path.dirname(in_file_path)
    train_file_path = in_file_pathAbs + '/{}train.csv'.format(name_of_class)
    test_file_path = in_file_pathAbs + '/{}test.csv'.format(name_of_class)
    train_file.to_csv(train_file_path, columns=columns, index=False)
    test_file.to_csv(test_file_path, columns=columns, index=False)

    # convert every row in  csv files to text file
    train_dataset = open(train_file_path, 'r')
    test_dataset = open(test_file_path, 'r')
    train_output_txt = os.path.join(in_file_pathAbs, 'train', name_of_class)
    test_output_txt = os.path.join(in_file_pathAbs, 'test', name_of_class)
    if not os.path.isdir(train_output_txt):
        os.makedirs(train_output_txt)
    if not os.path.isdir(test_output_txt):
        os.makedirs(test_output_txt)
    tra = 0
    tes = 0
    for row in csv.reader(train_dataset):
        out_file = train_output_txt + "/{}.txt".format(tra)
        with open(out_file, "w") as my_output_file:
            my_output_file.write("".join(row))
        my_output_file.close()
        tra = tra + 1
    train_dataset.close()
    for row in csv.reader(test_dataset):
        out_file = test_output_txt + "/{}.txt".format(tes)
        with open(out_file, "w") as my_output_file:
            my_output_file.write("".join(row))
        my_output_file.close()
        tes = tes + 1
    test_dataset.close()


def csv_to_txt_train_test_group_with_preprocess(in_file_path: str, name_of_class: str, columns: list):
    """this function to split the csv to 80% train,20% test next preprocess the line then convert every line to [index].txt
    parameters:
    inFilePath:string path of csv file
    nameOfClass:string name of classes
    columns:list string what column name you need to detect """
    # split the data to train 80% and test 20%
    in_file = pd.read_csv(in_file_path)
    in_file['split'] = np.random.randn(in_file.shape[0], 1)
    msk = np.random.rand(len(in_file)) <= 0.8
    train_file = in_file[msk]
    test_file = in_file[~msk]
    in_file_pathAbs = os.path.dirname(in_file_path)
    train_file_path = in_file_pathAbs + '/{}train.csv'.format(name_of_class)
    test_file_path = in_file_pathAbs + '/{}test.csv'.format(name_of_class)
    train_file.to_csv(train_file_path, columns=columns, index=False)
    test_file.to_csv(test_file_path, columns=columns, index=False)

    # convert every row in  csv files to text file
    train_dataset = open(train_file_path, 'r')
    test_dataset = open(test_file_path, 'r')
    train_output_txt = os.path.join(in_file_pathAbs, 'train', name_of_class)
    test_output_txt = os.path.join(in_file_pathAbs, 'test', name_of_class)
    if not os.path.isdir(train_output_txt):
        os.makedirs(train_output_txt)
    if not os.path.isdir(test_output_txt):
        os.makedirs(test_output_txt)
    tra = 0
    tes = 0
    for row in csv.reader(train_dataset):
        row_text = "".join(row)
        row_after_process = preprocess_text(row_text, '/home/ahmad/Desktop/classificationTextProject/testdata/arabic-stop')
        out_file = train_output_txt + "/{}.txt".format(tra)
        with open(out_file, "w") as my_output_file:
            my_output_file.write("".join(row_after_process))
        my_output_file.close()
        tra = tra + 1
    train_dataset.close()
    for row in csv.reader(test_dataset):
        row_text = "".join(row)
        row_after_process = preprocess_text(row_text, '/home/ahmad/Desktop/classificationTextProject/testdata/arabic-stop')
        out_file = test_output_txt + "/{}.txt".format(tes)
        with open(out_file, "w") as my_output_file:
            my_output_file.write("".join(row_after_process))
        my_output_file.close()
        tes = tes + 1
    test_dataset.close()


def csv_full_to_txt(files_dir: str, output_dir: str):
    """convert the full CSV files to full_text_{}.txt
    parameters:
    files_dir : srting path of CSV files
    out_dir:string path of path folder output"""
    for file in os.listdir(files_dir):
        if file.endswith('.csv'):
            file_csv_dir = "{}/{}".format(files_dir, file)
            text_csv = open(file_csv_dir, 'r')
            text_csv_read = text_csv.read()
            file_full_name = os.path.basename(file)
            file_name = os.path.splitext(file_full_name)[0]
            file_output_name = "{}/full_text_{}.txt".format(output_dir, file_name)
            with open(file_output_name, 'w') as out_file:
                out_file.write(text_csv_read)
            out_file.close()
