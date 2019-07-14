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

	"""Preprocess class	
	Attributes:
	    counter (list): Description
	    unit (int): Description
	"""
	
	def __init__(self):
		self.unit = 0
		self.counter = [['unit',0]]

	def localLoad(self):
		df = pd.read_excel("data/mem30s.xlsx")
		print(df.head())
		#Further preprocessing the df file
	def unzipFile(zipped,unzipped):
		print('unzipFile {} to {}'.format(zipped, unzipped))
		with bz2.BZ2File(zipped, 'rb') as file:
			with open(unzipped, 'wb') as new_file:
				for data in iter(lambda: file.read(100 * 1024), b''):
					new_file.write(data)
	def toggleDownload(self,dataDir,fileName, url):
		# Create directory if not initiated
		if not os.path.exists(dataDir):
			print("File not found, creating directory as %s" % dataDir)
			os.mkdir(dataDir)
		filePathNobz2 = os.path.join(dataDir, fileName)
		filePath = filePathNobz2 + '.bz2'
		if not os.path.exists(filePathNobz2):
			if not os.path.exists(filePath):
				print("Downloading %s to %s" % (url, filePath))
				filePath, _ = urllib.request.urlretrieve(url, filePath)
				statinfo = os.stat(filePath)
				print("Successfully downloaded", fileName, statinfo.st_size, "bytes")
				decompress_file(filePath, filePathNobz2)
				return filePathNobz2
			else:
				decompress_file(filePath, filePathNobz2)
				return filePathNobz2
		else:
			print("{} already exists".format(filePathNobz2))
			return filePathNobz2
	def getAsus(fileName):
		asusSet=set()
		with open(fileName,'r') as f:
			for line in f:
				splits = line.strip().strip(",")
				if splits[0] in asusSet:
					pass
				else:
					asusSet.add(splits[0])
		pass
		return len(asusSet),list(asusSet)

	def randBatchGen(x,y,batchSize):
		while True:
			sampleID = sample(list(np.arange(len(x))),batchSize)
			yield x[sampleID].T, y[sampleID].T

	def decode(sequence,lookup,seperator=''):
		return separator.join([lookup[element] for element in sequence if element])







if __name__ == "__main__":
	pass
else:
	print("Module used:",__name__)