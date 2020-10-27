import csv
import pandas as pd
import numpy as np
import os


def csv_to_txt_train_test_group(inFilePath: str, nameOfClass: str, columns: list):
    """this function to split the csv to 80% train,20% test then convert every line to [index].txt
    parameters:
    inFilePath:string path of csv file
    nameOfClass:string name of classes
    columns:list string what column name you need to detect """
    # split the data to train 80% and test 20%
    inFile = pd.read_csv(inFilePath)
    inFile['split'] = np.random.randn(inFile.shape[0], 1)
    msk = np.random.rand(len(inFile)) <= 0.8
    trainFile = inFile[msk]
    testFile = inFile[~msk]
    inFilePathAbs = os.path.dirname(inFilePath)
    trainFilePath = inFilePathAbs + '/{}train.csv'.format(nameOfClass)
    testFilePath = inFilePathAbs + '/{}test.csv'.format(nameOfClass)
    trainFile.to_csv(trainFilePath, columns=columns, index=False)
    testFile.to_csv(testFilePath, columns=columns, index=False)

    # convert every row in  csv files to text file
    trainDataset = open(trainFilePath, 'r')
    testDataset = open(testFilePath, 'r')
    trainOutputtxt = os.path.join(inFilePathAbs, 'train', nameOfClass)
    testOutputtxt = os.path.join(inFilePathAbs, 'test', nameOfClass)
    if not os.path.isdir(trainOutputtxt):
        os.makedirs(trainOutputtxt)
    if not os.path.isdir(testOutputtxt):
        os.makedirs(testOutputtxt)
    tra = 0
    tes = 0
    for row in csv.reader(trainDataset):
        outfile = trainOutputtxt + "/{}.txt".format(tra)
        with open(outfile, "w") as my_output_file:
            my_output_file.write("".join(row))
        my_output_file.close()
        tra = tra + 1
    trainDataset.close()
    for row in csv.reader(testDataset):
        outfile = testOutputtxt + "/{}.txt".format(tes)
        with open(outfile, "w") as my_output_file:
            my_output_file.write("".join(row))
        my_output_file.close()
        tes = tes + 1
    testDataset.close()


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
