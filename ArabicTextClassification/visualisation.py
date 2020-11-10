import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud import WordCloud
import os
import re
import matplotlib.pyplot as plt


def removeWeirdChars(text):
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


def visualisation_vocabulary_folder(files_dir: str, status_of_preprocess: bool, output_dir_images: str):
    """this method visulisation all files in folder but .txt
    parameter:
    files_dir:string of directory the files you need to visualisation_vocabulary,
    status_of_preprocess: boolean value:true=>after,false=>before
    output_dir_images : string of path the output direction you need to save the images """
    if status_of_preprocess:
        preprocess = 'after_preprocess'
    else:
        preprocess = 'befor_preprocess'
    for file in os.listdir(files_dir):
        if file.endswith('.txt'):
            file_full_name = os.path.basename(file)
            file_name = os.path.splitext(file_full_name)[0]
            print("start : {}".format(file_name))
            text_file_bath = os.path.join(files_dir, file)
            text_file = open(text_file_bath, 'r')
            text = text_file.read()
            if not status_of_preprocess:
                text = removeWeirdChars(text)
                text = arabic_reshaper.reshape(text)
                text = get_display(text)
            else:
                text = arabic_reshaper.reshape(text)
                text = get_display(text)
            word_cloud = WordCloud(width=1900, height=800, font_path='NotoNaskhArabic-Regular.ttf').generate(text)
            output_image = "{}/{}_{}.png".format(output_dir_images, preprocess, file_name)
            word_cloud.to_file(output_image)
            print("finish : {}".format(file_name))
            text_file.close()


def visualisation_vocabulary_file(text: str, output_dir: str, name_of_output_file: str):
    """this function visualisation file
    parameter:
    text: string text
    output_dir: string of path you need save image
    name_of_output_file: string of name the image"""

    textArabic = arabic_reshaper.reshape(text)
    textArabic = get_display(textArabic)
    word_cloud = WordCloud(font_path='NotoNaskhArabic-Regular.ttf').generate(textArabic)
    output_image = "{}/{}.png".format(output_dir, name_of_output_file)
    word_cloud.to_file(output_image)


def plot_graphs(history, string, name_image):
    name = "{}.png".format(name_image)
    fig = plt.figure()
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    fig.savefig(name)


