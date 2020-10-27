import os
from nltk.tokenize import RegexpTokenizer
import nltk


def remove_punctuation(text: str):
    """this function to remove the punctuation
    parameter:
    text:string """
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(text))


def remove_stop_words(text: str, stopwords: list):
    """this function to remove the stop words
    parameter:
    text:string
    stopwords : list of string words you need deleted"""
    text_tokens = nltk.tokenize.wordpunct_tokenize(text)
    return " ".join([word for word in text_tokens if not word in stopwords])


def preprocess_text(in_files_dir: str, out_files_dir: str, stopwords: list):
    """this function to remove the stop words and punctuation
    parameter:
    in_file_dir:string path input text files
    out_files_dir:string path output text files
    stopwords : list of string words you need deleted"""
    for file in os.listdir(in_files_dir):
        if file.endswith(".txt"):
            file_dir = "{}/{}".format(in_files_dir, file)
            text_file = open(file_dir, 'r')
            text_file = text_file.read()
            nostop = remove_stop_words(text_file, stopwords)
            nopunctuation = remove_punctuation(nostop)
            file_full_name = os.path.basename(file)
            file_name = os.path.splitext(file_full_name)[0]
            out_file_txt = '{}/preprocess_{}.txt'.format(out_files_dir, file_name)
            with open(out_file_txt, 'w') as output_file:
                output_file.write(nopunctuation)
            output_file.close()


def stop_words(file_path: str):
    """this function to detect the stop word file
    parameter:
    file_path: string the wordStop file path"""
    file = open(file_path, 'r', encoding='utf-8')
    stopwords_arabic = file.read().splitlines()
    return stopwords_arabic
