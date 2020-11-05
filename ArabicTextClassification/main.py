from ithelpClassification.csv_to_txt import csv_full_to_txt, csv_to_txt_train_test_group_with_preprocess
from ithelpClassification.visualisation import visualisation_vocabulary_folder
from ithelpClassification.read_csv_spilt_row_by_language import csv_spilt_to_CSVs_by_language
from ithelpClassification.prprocessArabicText import preprocess_text_in_folder
import os

# 1)
# spilt all csv files to csv files by row language:
#############################################################################################
# csv_files_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/csvFiles"
# ar_csv_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/classesArabicCSV"
# en_csv_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/classesEnglishCSV"
# other_csv_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/classesOtherCSV"
# blackboard_csv_file = "{}/blackboard.csv".format(csv_files_dir)
# helpdesk_support_csv_file = "{}/helpdesk_support.csv".format(csv_files_dir)
# learning_tech_resource_csv_file = "{}/learning_tech_resource.csv".format(csv_files_dir)
# main_gate_university_csv_file = "{}/main_gate_university.csv".format(csv_files_dir)
# maward_csv_file = "{}/maward.csv".format(csv_files_dir)
# network_security_csv_file = "{}/network_security.csv".format(csv_files_dir)
# other_csv_file = "{}/other.csv".format(csv_files_dir)
# passive_csv_file = "{}/passive.csv".format(csv_files_dir)
# scientific_research_system_csv_file = "{}/scientific_research_system.csv".format(csv_files_dir)
# sis_csv_file = "{}/sis.csv".format(csv_files_dir)
# system_infrastructure_apps_csv_file = "{}/systems_infrastructure_apps.csv".format(csv_files_dir)
# telephone_conferences_csv_file = "{}/telephone_conferences.csv".format(csv_files_dir)
# transaction_flow_system_csv_file = "{}/transaction_flow_system.csv".format(csv_files_dir)
#
# csv_spilt_to_CSVs_by_language(blackboard_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(helpdesk_support_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(learning_tech_resource_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(main_gate_university_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(maward_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(network_security_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(other_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(passive_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(scientific_research_system_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(sis_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(system_infrastructure_apps_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(telephone_conferences_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
# csv_spilt_to_CSVs_by_language(transaction_flow_system_csv_file, ar_csv_dir, en_csv_dir, other_csv_dir)
####################################################################################################################

# 2)
# convert the Arabic_full-csv files to full-txt files without preprocess:

####################################################################################################################
# csv_folder_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/classesArabicCSV"
# txt_folder_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/fullTextsArabic"
# csv_full_to_txt(csv_folder_dir, txt_folder_dir)
####################################################################################################################

# 3)
# visualisation all text files befor the preprocess:
# txt_folder_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/fullTextsArabic"
# image_folder_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/image_visu_befor_prepro"

# 3.1)
# visualisation the text befor the preprocess

# visualisation_vocabulary_folder(txt_folder_dir, False, image_folder_dir)

# 3.2)
# visualisation the text after the preprocess

# txt_folder_preprocess_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/fulltextsArabic_preprocess"
# arabic_stop_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/arabic-stop"
# image_folder_preprocess_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/image_visu_after_prepro"
# preprocess_text_in_folder(txt_folder_dir, txt_folder_preprocess_dir, arabic_stop_dir)
# visualisation_vocabulary_folder(txt_folder_preprocess_dir, True, image_folder_preprocess_dir)

# 4)
# convert the all csv file to row_txt_files with preprocess:
# csv_files_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/classesArabicCSV"
# stop_word_dir = "/home/ahmad/Desktop/UniversityChatbotTextClassification/ithelpClassification/arabic-stop"
# for file in os.listdir(csv_files_dir):
#     if file.endswith('.csv'):
#         file_full_name = os.path.basename(file)
#         file_name = os.path.splitext(file_full_name)[0]
#         file_dir = "{}/{}".format(csv_files_dir, file_full_name)
#         csv_to_txt_train_test_group_with_preprocess(file_dir, file_name, ['description'], stop_word_dir)


# 5)
# build the model: in build_model.py
