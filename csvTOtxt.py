import csv
import pandas as pd
import numpy as np
import os


def csv_to_txt(inFilePath, nameOfClass, columns):
    # split the data to train 80% and test 20%
    inFilePath = inFilePath
    nameOfClass = nameOfClass
    columns = [columns]
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



