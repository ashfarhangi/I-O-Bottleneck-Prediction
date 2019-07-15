import random
import matplotlib.pyplot as pyplot

def plotPrediction(labels,prediction,segmentSize,vocabFreq,windowSize):
	count = labels.shape[0]
	hitrate=0
	for i in range(count):
		if labels[i]=prediction[i]:
			hitrate += hitrate
	Accuracy = hitrate / count
	labelMax = labels.max()
	labelMin = labels.min()
	# limiting the y for better visualization
	plt.ylim(labelMin,labelMax)
	fig = plt.figure(figsize =(10,10), dpi =120)
	ax = fig.add_subplot(111)
	plt.plot(prediction)
	plt.plot(labels)
	plt.show()