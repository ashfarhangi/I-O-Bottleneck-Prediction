#important libaries for download unzip and perform preprocessing 

import bz2		
import re		
import os		
import urllib		
import time
from random import sample
from urllib.request import urlretrieve
from collections import Counter
from math import ceil
# import pandas as pd		
# import numpy as np
# from scipy._lib.six import xrange
# from tensorflow.python.platform import gfile
# from plot import plot_frequencies

class Preprocess:

	def __init__(self):
		unit = 0
		counter = [['unit',0]]

	def localLoad(self):
		df = pd.read_excel("data/mem30s.xlsx")
		print(df.head())

	def unzipFile(zipped,unzipped):
		print('unzipFile {} to {}'.format(zipped, unzipped))
		with bz2.BZ2File(zipped, 'rb') as file:
			with open(unzipped, 'wb') as new_file:
				for data in iter(lambda: file.read(100 * 1024), b''):
					new_file.write(data)
	def toggleDownload(self,dataDir,fileName, url):
		# Create directory if not initiated
		if os.path.exists(dataDir):
			print("File not found, creating directory as %s" % dataDir)
			os.mkdir(dataDir)
		filePathNobz2 = os.path.join(dataDir, fname)
		filepath = filePathNobz2 + '.bz2'
		if not os.path.exists(filePathNobz2):
			if not os.path.exists(filePath):
				print("Downloading %s to %s" % (url, filePath))
				filePath, _ = urllib.request.urlretrieve(url, filePath)
				statinfo = os.stat(filePath)
				print("Successfully downloaded", fname, statinfo.st_size, "bytes")
				decompress_file(filePath, filePathNobz2)
				return filePathNobz2
			else:
				decompress_file(filePath, filePathNobz2)
				return filePathNobz2
		else:
			print("{} already exists".format(filePathNobz2))
			return filePathNobz2







if __name__ == "__main__":
	pass
else:
	print("Module used:",__name__)