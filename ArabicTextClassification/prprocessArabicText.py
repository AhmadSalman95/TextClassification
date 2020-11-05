import os
from nltk.tokenize import RegexpTokenizer
import nltk
import re

def removeWeirdChars(text: str):
    """this function remove the weird chars
    :parameter
    text : string of text you needed to remove the wired chars """
    weridPatterns = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)
    return weridPatterns.sub(r'', text)


def remove_punctuation(text: str):
    """this function to remove the punctuation
    parameter:
    text:string of the text """
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(text))


def remove_stop_words(text: str, stopwords: list):
    """this function to remove the stop words
    parameter:
    text:string of the text
    stopwords : list of string words you need deleted"""
    text_tokens = nltk.tokenize.wordpunct_tokenize(text)
    return " ".join([word for word in text_tokens if not word in stopwords])


def preprocess_text_in_folder(in_files_dir: str, out_files_dir: str, stop_words_dir: str):
    """this function to remove the stop words and punctuation and save the output to .txt file
    parameter:
    in_file_dir:string path input text files
    out_files_dir:string path output text files
    stopwords :string of path words you need deleted"""
    stop_words_list = stop_words(stop_words_dir)
    for file in os.listdir(in_files_dir):
        if file.endswith(".txt"):
            file_dir = "{}/{}".format(in_files_dir, file)
            text_file = open(file_dir, 'r')
            text_file = text_file.read()
            noWeirdChars = removeWeirdChars(text_file)
            nostop = remove_stop_words(noWeirdChars, stop_words_list)
            nopunctuation = remove_punctuation(nostop)
            file_full_name = os.path.basename(file)
            file_name = os.path.splitext(file_full_name)[0]
            out_file_txt = '{}/preprocess_{}.txt'.format(out_files_dir, file_name)
            with open(out_file_txt, 'w') as output_file:
                output_file.write(nopunctuation)
            output_file.close()


def preprocess_text(text: str, stop_words_dir: str):
    """this method remove stop words and the punctuation and return the text
     parameter:
     text: string of text
     stop_word_dir: string of file stop words bath"""
    stop_words_list = stop_words(stop_words_dir)
    noWeirdChars = removeWeirdChars(text)
    no_stop = remove_stop_words(noWeirdChars, stop_words_list)
    no_punctuation = remove_punctuation(no_stop)
    return no_punctuation


def stop_words(file_path: str):
    """this function to detect the stop word file
    parameter:
    file_path: string the wordStop file path"""
    file = open(file_path, 'r', encoding='utf-8')
    stopwords_arabic = file.read().splitlines()
    return stopwords_arabic
