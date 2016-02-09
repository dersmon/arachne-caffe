import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import modules.arachne_caffe as ac
import modules.arachne_KNN_prediction as aknnp
import modules.arachne_KMeans_prediction as akmp

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

saveFile = "./clusters_factor_2000.npy"

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
	factor = (len(training) / 2000)
	if factor < 1:
		factor = 1

	clusterCount = labelCount * 0.5


	clusters = None
	clusters = akmp.kMeans(training, clusterCount, 150)

	print 'Writing file ' + saveFile
	if not os.path.exists(os.path.dirname(saveFile)):
		os.makedirs(os.path.dirname(saveFile))
	with open(saveFile, "w") as outputFile:
		np.save(outputFile, clusters)

	# with open("clusters.npy", 'r') as inputFile:
	# 	clusters = np.load(inputFile)

	clusters = akmp.clusterAnalysis(clusters, training, labelCount)

	print 'Writing file ' + saveFile
	if not os.path.exists(os.path.dirname(saveFile)):
		os.makedirs(os.path.dirname(saveFile))
	with open(saveFile, "w") as outputFile:
		np.save(outputFile, clusters)


def testKMeans(test):

	with open(saveFile, 'r') as inputFile:
		clusters = np.load(inputFile)
		akmp.clusterTest(clusters, test)

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

labelCount = len(trainingActivations[0][4096:])

calculateKMeans(trainingActivations, labelCount)

if testActivationsPath.endswith('.npy'):
	testActivations = ac.activationsFromFile(testActivationsPath)
else:
	print(testActivations + " does not seem to be a npy-file with activations.")

testKMeans(testActivations)
