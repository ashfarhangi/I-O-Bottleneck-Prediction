#important libaries for download unzip and perform preprocessing 

import pandas as pd		
import bz2		
import re		
import os		
import urllib		
import numpy as np		
import time
from random import sample
from urllib.request import urlretrieve
from collections import Counter
from math import ceil
from scipy._lib.six import xrange
from tensorflow.python.platform import gfile
from plot import plot_frequencies


class Preprocess:

	def __init__(self):

	def localLoad(self):
		df = pd.read_excel("data/mem30s.xlsx")
		print(df.head())
	def toggleDownload(self):



if __name__ == "__main__":
	pass
else:
	print("Module used:",__name__)