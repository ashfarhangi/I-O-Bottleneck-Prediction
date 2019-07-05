# Famous quotes: All code is garbage
# Functionality over beauty

# This is our main class where he handle the preprocessing, prediciton in one class
# The goal is to have only two modules for preprocessing and one prediction module for the use of seq2seq model
"""Here is the setup"""
# Importing the packages
# importing the model for seq2seq














import numpy as np
import preprocess
import method
import os
import re

class Main:
	
	def __init__(self):
		# Lets initiate the values that are necessacry for our model.
		url = 'http://skuld.cs.umass.edu/traces/storage/Financial1.spc.bz2'
		self.boolWhile = True
		self.vocabFreq=5
		self.batchSize=512
		self.epochs=100
		self.segmentSize=1024
		self.numLayers=5
		#64 milli seconds
		self.windowSize=64
		self.trainMode =False
		self.fileName='Financial1.spc'
	def main(self):
		while self.boolWhile:
			x.predict()
		self.boolWhile = False
	def download(self):
		toggleDownload(dataDir='data',fileName=fileName, url = url)
	def predict(self):
		print('predicting')
		df = preprocess.df
		print(df.head())
		
if __name__ == "__main__":
	x = Main()
	y = Preprocess()
	z = Method() 
	x.main()
else:
	print("module used:", __name__)