import subprocess
import sys


def install(package: str):
    """this function install all packages the project needed
    parameter:
    package : string of name the package you needed to install"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("finish install this package ", package, "\n")

def upgrade(package : str):
    """this function install upgrade all packages the project needed
    parameter:
    package : string of name the package you needed to upgrade"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
    print("finish upgrade this package ", package, "\n")


# upgrade("pip")
# install("pandas")
# install("numpy")
# install("nltk")
# install("langdetect")
# install("python-csv")
# install("arabic_reshaper")
# install("python-bidi")
# install("wordcloud")
# install("tensorflow>=2.0.0")
# upgrade("tensorflow")
# install("pyLDAvis")
# install("IPython")
# install("eli5")
