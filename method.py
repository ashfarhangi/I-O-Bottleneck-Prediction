# import libaries including tf 14
# import tensorflow as tf
# import numpy as np
import sys
# import time

class sec2sec(object):
	def __init__(self,
				trainSize,
				batchSize,
				sequenceXLength,
				sequenceYLength,
				vocabXSize,
				vocabYSize,
				checkPointPath,
				epochs,
				embeddedDim,
				methodName,
				numLayers,
				lr=0.0001):
		print("started")
		self.trainSize=trainXCount
		self.batchSize=batchSize
		self.sequenceXLength=inputLength
		self.sequenceYLength=labelLength
		self.checkPointPath = checkPointPath
		self.epochs=epochs,
		self.methodName=methodName
		sys.stdout.write("Starting the graph...")
		#Added during debugging
		setattr(tf.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
		setattr(tf.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)
		def __graph__():
			tf.reset_default_graph()

			self.inputEnc = [ tf.placeholder(shape=[None,],dtype=tf.init64, name='ie_{}'.format(t)) for t in range(sequenceXLength)]
			self.labels = [tf.placeholder(shape =[None,],dtype=tf.init64,name="ie_{}".format(t)) for t in range(sequenceYLength)]
			self.inputDec = 

if __name__ == "__main__":
	print("nothing to show here")
	print("Run the main.py")
else:
	print("Module used:",__name__)