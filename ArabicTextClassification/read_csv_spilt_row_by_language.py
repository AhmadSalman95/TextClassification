import langdetect
import csv
import os


def append_row(file_name: str, row: str):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(row)


def csv_spilt_to_CSVs_by_language(file_dir: str, out_dir_ar: str, out_dir_en: str, out_dir_other: str):
    """this function spilt the csv file to csv files by row language
    :parameter
    file_dir: string of path csv file you want spilt
    out_dir_ar: string of path csv file the language is arabic
    out_dir_en: string of path csv file the language is english
    out_dir_other : string of path csv file the language doesn't detect"""
    file_name = os.path.basename(file_dir)
    file_name = os.path.splitext(file_name)[0]
    file = open(file_dir, 'r')
    en_row = 0
    ar_row = 0
    other_row = 0
    file_row = 0
    mistake_row = 0
    arabic_file = "{}/{}.csv".format(out_dir_ar, file_name)
    english_file = "{}/{}.csv".format(out_dir_en, file_name)
    other_file = "{}/{}.csv".format(out_dir_other, file_name)

    for row in csv.reader(file):
        try:
            lan_row = langdetect.detect(row[0])
        except:
            file_row = file_row + 1
            print("{}: detect row mistake: #{}".format(file_name, file_row))
            mistake_row = mistake_row + 1
        if lan_row == 'en':
            append_row(english_file, row)
            en_row = en_row + 1
        elif lan_row == 'ar':
            append_row(arabic_file, row)
            ar_row = ar_row + 1
        else:
            append_row(other_file, row)
            other_row = other_row + 1
        file_row = file_row + 1
        # print("file row : ", file_row)

    print("the {} rows is: \n en-row = {} \n ar-row = {} \n other-row = {} \n mistake-row = {} \n".format(file_name,
                                                                                                          en_row,
                                                                                                          ar_row,
                                                                                                          other_row,
                                                                                                          mistake_row))
