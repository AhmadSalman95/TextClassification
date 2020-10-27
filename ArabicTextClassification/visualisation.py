import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud import WordCloud
import os


def visualisation_vocabulary_txt(files_dir: str, status_of_preprocess: bool):
    """parameter:
    files_dir:string of directory the files you need to visualisation_vocabulary,
    status_of_preprocess: boolean value:true=>after,false=>before"""
    if status_of_preprocess:
        preprocess = 'after_preprocess'
    else:
        preprocess = 'befor_preprocess'
    for file in os.listdir(files_dir):
        if file.endswith('.txt'):
            text_file_bath = os.path.join(files_dir, file)
            text_file = open(text_file_bath, 'r')
            text = text_file.read()
            text = arabic_reshaper.reshape(text)
            text = get_display(text)
            word_cloude = WordCloud(font_path='NotoNaskhArabic-Regular.ttf').generate(text)
            file_full_name = os.path.basename(file)
            file_name = os.path.splitext(file_full_name)[0]
            output_image = "{}/{}_{}.png".format(files_dir, preprocess, file_name)
            word_cloude.to_file(output_image)
            text_file.close()