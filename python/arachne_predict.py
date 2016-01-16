import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import modules.arachne_caffe as ac
import modules.arachne_KNN_prediction as aknnp
import modules.arachne_KMeans_prediction as akmp

def drawVectors(vectors):
	labels = np.array(vectors)[:,4096:]
	grid = np.array(vectors)[:,0:4096]

	maxValue = np.amax(grid)
	minValue = np.amin(grid)

	# print 'max activation: ' + str(maxValue) + ', min activation: ' + str(minValue)

	scaled = grid * (255 / maxValue)
	#
	# scaled = scaled.reshape(scaled.shape[0],4,1024)
	# scaled = scaled.mean(axis=1)

	plt.imshow(scaled, 'Greys_r', interpolation='none')
	plt.show()

def testKNN(training, test, labelCount):
	filePath = "./neighbours_elastic.npy"
	testNeighbours = aknnp.nearestNeighbours(training, test, 50)

	print 'Writing file ' + filePath
	if not os.path.exists(os.path.dirname(filePath)):
		os.makedirs(os.path.dirname(filePath))
	with open(filePath, "w") as outputFile:
		np.save(outputFile, testNeighbours)

	# with open(filePath, 'r') as inputFile:
	# 	testNeighbours = np.load(inputFile)

	count = 0
	data = []
	while count < 50 and count < len(training):

		(correct, wrong, correctPerLabel, wrongPerLabel) = aknnp.kNearestAnalysed(testNeighbours, count + 1, labelCount)

		data.append([count, float(correct)/(wrong + correct)])
		count += 1
		if((count + 1) % 100 == 0):
			print 'Calculated prediction for ' + str(count + 1) + ' nearest neighbours.'

		print str(correct) + ", " + str(wrong) + ", " + str(float(correct)/(wrong + correct))

	print 'Calculated prediction for ' + str(count) + ' nearest neighbours.'
	data = np.array(data)

	plt.plot(data[:,0], data[:,1], 'k')
	plt.axis([1, len(data), 0, 1])
	plt.grid(True)
	plt.show()

def calculateKMeans(training, labelCount):
	clusters = akmp.multipleLabelsPerImage(training, labelCount * 2, 50)

	filePath = "./clusters_small.npy"

	print 'Writing file ' + filePath
	if not os.path.exists(os.path.dirname(filePath)):
		os.makedirs(os.path.dirname(filePath))
	with open(filePath, "w") as outputFile:
		np.save(outputFile, clusters)

	clusters = akmp.clusterAnalysisMultipleLabels(clusters, training)

def testKMeans(test, labelCount):

	with open("clusters_small.npy", 'r') as inputFile:
		clusters = np.load(inputFile)

		akmp.clusterTest(clusters, test, labelCount)

trainingActivationsPath = ""
testActivationsPath = ""

trainingActivations = None
testActivations = None

if(len(sys.argv) < 2):
	print("No activation vectors provided.")
else:
	trainingActivationsPath = sys.argv[1]

if(len(sys.argv) < 3):
	print("No activation vectors provided.")
else:
	testActivationsPath = sys.argv[2]

if trainingActivationsPath.endswith('.npy'):
	trainingActivations = ac.activationsFromFile(trainingActivationsPath)
else:
	print(trainingActivationsPath + " does not seem to be a npy-file with activations.")

if testActivationsPath.endswith('.npy'):
	testActivations = ac.activationsFromFile(testActivationsPath)
else:
	print(testActivations + " does not seem to be a npy-file with activations.")

labelCount = len(trainingActivations[0][4096:])

#drawVectors(trainingActivationVectors)

#testKNN(trainingActivations, testActivations, labelCount)

#calculateKMeans(trainingActivations, labelCount)
testKMeans(testActivations, labelCount )
