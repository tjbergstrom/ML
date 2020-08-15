# train.py
# August 2020
# Just rying out a sentiment analysis exercise

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
import os
from nltk.corpus import stopwords
import string


# Pre-Processing

# Remove "@user"
# Comment out because only needs to be done once
#os.chdir("processed_data")
#os.system("sed -i 's/'@user'/''/g' train.csv")




