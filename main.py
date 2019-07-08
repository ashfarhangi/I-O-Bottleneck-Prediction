# Author: github.com/ashfarhangi
# TL-DR: this is the main py file where we handle the preprocessing, prediciton model,
 # and plot customized for a specific project
# 1.import file
# 2.call preprocess methods for indexing the sectors (encode,decode)
# 3.call prediciton model for start on sess 
# 4. predict or train based on if statement and checkPoint 
# 5. plot

# Description: This is our main class where he handle the preprocessing, prediciton in one class
# The goal is to have only two modules for preprocessing and one prediction module for the use of seq2seq model

"""Here is the setup"""
# Importing the packages
import os
import re
# import numpy as np

# importing the model for seq2seq

from preprocess import Preprocess
from method import Method
# importing packages
# importing seq model.py, plot.py preprocess.py
class Main:

	"""The main class where it handles two packages
	
	Attributes:
	    batchSize (int): Description
	    boolWhile (bool): Description
	    epochs (int): Description
	    fileName (str): Description
	    numLayers (int): Description
	    segmentSize (int): Description
	    trainMode (bool): Description
	    url (str): Description
	    vocabFreq (int): Description
	    windowSize (int): Description
	"""
	
	def __init__(self):
# <Preprocess>
# download link
# hyperparameters : batch, epochs and more
# toggle download<pre> 
# Lets initiate the values that are necessacry for our model.
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
		self.url = 'http://skuld.cs.umass.edu/traces/storage/Financial1.spc.bz2'
	
	def main(self):
		
		while self.boolWhile:
			x.preprocess()
			x.predict()
			self.boolWhile = False
	
	def preprocess(self):

		y.toggleDownload(dataDir='data',fileName=self.fileName, url = self.url)
		_, sector2index,index2sector,_ =y.prepareSectorSequence(dataDir="data",
			trace=self.fileName,vocabFreq=vocabFreq,windowSize=windowSize,
			sectorSize = sectorSize)
		sequenceName = os.path.splitext(self.fileName)[0]+".sequences"+str(vocabFreq)+"ss-"+str(
			segmentSize)+"ws-"+str(windowSize)
		trainXPath, trainYPath,testXPath,testYPath = y.prepareSectorData("data",sequenceName,sector2index,trainTestSplit = 0.1,token=None)
		print(trainXPath,trainYPath,testXPath,testYPath)
		(trainX,trainY),(testX,testY) = getTrainTest(trainXPath, trainYPath,testXPath,testYPath)
		print("train input size:{} train label size:{} \n test input size:{} test label size:{}".format(trainX.shape,trainY.shape,testX.shape,testY.shape))
		inputLength = trainX.shape[-1]
		labelLength = trainY.shape[-1]
		inputVocabSize = len(sector2index)
		labelVocabSize = len(sector2index)
		del sector2index
		#Free memory
		methodName = os.path.splitext(self.fileName)[0]+"sectorSeqMethod"
		trainXCount=trainX.shape[0]
		print("train count: {}".format(trainX.shape[0]))
		print("test count: {}".format(testX.shape[0]))
		batchTrainGenerate= y.batchRandomGenerate(trainX,trainY,batchSize)
		batchValidationGenerate= y.batchRandomGenerate(testX,testY,batchSize)
		#fin
	def predict(self):

		network= z.sec2sec(
			trainSize=trainXCount,
			batchSize=batchSize,
			sequenceXLength=inputLength,
			sequenceYLength=labelLength,
			vocabXSize=inputVocabSize,
			vocabYSize=labelVocabSize,
			checkPointPath = 'checkpoint/',
			epochs=epochs,
			embeddedDim=embeddedDim,
			methodName=methodName,
			numLayers=numLayers)
		sess = network.restoreLastcheckPoint()
		if trainmode == True:
			sess = network.train(batchTrainGenerate,batchValidationGenerate)
		else:
			input_,labels_ = batchValidationGenerate.__next__()
			output = network.predict(sess,input_)

			replies = []
			label = list()
			prediciton = list()

			for ii, il, oi in zip(input_.T, labels_.T, output):
				q = decode(sequence=ii, lookup=idx2block, separator=' ')
				l = decode(sequence=il, lookup=idx2block, separator=' ')
				o = decode(sequence=oi, lookup=idx2block, separator=' ')
				decoded = o.split(' ')

				if decoded.count('unit') == 0:
					if decoded not in replies:
						if len(l) == len(o):
							print('i: [{0}]\na: [{1}]\np: [{2}]\n'.format(q, l, ' '.join(decoded)))
							print("{}".format("".join(["-" for i in range(80)])))
							lsplits = l.split()
							osplits = o.split()
							for lspl in lsplits:
								match = re.match(r"(\d+)(\w)", lspl)
								block, iotype = match.group(1), match.group(2)
								lbls.append(block)
							for osp in osplits:
								match = re.match(r"(\d+)(\w)", osp)
								block, iotype = match.group(1), match.group(2)
								preds.append(block)
							replies.append(decoded)
		def plot(self):
			label= np.asarray(label,dtype=np.int64)
			prediciton=np.asarray(prediciton,dtype=np.int64)
			p.plotPrediction(label,prediciton,segmentSize,vocabFreq,windowSize)	
if __name__ == "__main__":
	x = Main()
	y = Preprocess()
	z = Method()
	# p = Plot()
	x.main()
else:
	print("module used:", __name__)
