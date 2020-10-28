import subprocess
import sys


def install(package: str):
    """this function install all packages the project needed
    parameter:
    package : string of name the package you needed"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("finish install this package ", package, "\n")


install("pandas")
install("numpy")
install("nltk")
install("langdetect")
install("python-csv")
install("arabic_reshaper")
install("python-bidi")
install("wordcloud")
