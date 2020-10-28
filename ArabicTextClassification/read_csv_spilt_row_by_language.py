import langdetect
import csv


def append_row(file_name: str, row: str):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(row)


file_dir = ''
out_dir_ar = ''
out_dir_en = ''
out_dir_other = ''
file = open(file_dir, 'r')
en_row = 0
ar_row = 0
other_row = 0
file_row = 0
mistake_row = 0

for row in csv.reader(file):
    try:
        lan_row = langdetect.detect(row[0])
    except:
        file_row = file_row + 1
        print("detect row mistake: #", file_row)
        mistake_row = mistake_row+1
    if lan_row == 'en':
        append_row(out_dir_en, row)
        en_row = en_row + 1
    elif lan_row == 'ar':
        append_row(out_dir_ar, row)
        ar_row = ar_row + 1
    else:
        append_row(out_dir_other, row)
        other_row = other_row + 1
    file_row = file_row + 1
    print("file row : ", file_row)

print("the rows is: \n en-row = {} \n ar-row = {} \n other-row = {} \n mistake-row ".format(en_row,
                                                                                            ar_row,
                                                                                            other_row,
                                                                                            mistake_row))
