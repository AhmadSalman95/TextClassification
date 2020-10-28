import langdetect
import csv


def append_row(file_name: str, row: str):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(row)


file_dir = '/home/ahmad/Desktop/UniversityChatbotTextClassification/three-class-classification/BB-description.csv'
out_dir_ar = '/home/ahmad/Desktop/UniversityChatbotTextClassification/three-class-classification/BB-ar-row.csv'
out_dir_en = '/home/ahmad/Desktop/UniversityChatbotTextClassification/three-class-classification/BB-en-row.csv'
out_dir_other = '/home/ahmad/Desktop/UniversityChatbotTextClassification/three-class-classification/BB-other-row.csv'
file = open(file_dir, 'r')
en_row = 0
ar_row = 0
other_row = 0
file_row = 0

for row in csv.reader(file):
    try:
        lan_row = langdetect.detect(row[0])
    except:
        file_row = file_row+1
        print("dectect row mistack: #", file_row)
    if lan_row == 'en':
        append_row(out_dir_en, row)
        # with open(out_dir_en, 'w') as output_file:
        #     writer = csv.writer(output_file)
        #     writer.writerow(row)
        # # output_file.close()
        en_row = en_row + 1
    elif lan_row == 'ar':
        append_row(out_dir_ar, row)
        # with open(out_dir_ar, 'w') as output_file:
        #     writer = csv.writer(output_file)
        #     writer.writerow(row)
        # output_file.close()
        ar_row = ar_row + 1
    else:
        append_row(out_dir_other, row)
        # with open(out_dir_other, 'w') as output_file:
        #     writer = csv.writer(output_file)
        #     writer.writerow(row)
        # output_file.close()
        other_row = other_row + 1
    file_row = file_row + 1
    print("file row : ", file_row)

print("the rows is: \n en-row = {} \n ar-row = {} \n other-row = {} ".format(en_row, ar_row, other_row))
