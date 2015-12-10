import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import modules.arachne_caffe as anl
import modules.arachne_KNNPrediction as aknnp
import modules.arachne_KMeansPrediction as akmp



'''
def drawGrids(activations):

	global overallMaxValue
	#plt.imshow(grid[1], 'Greys_r', interpolation='none')
	#plt.show()

	counter = 0
	grids = []
	for activation in activations:

		grid =  np.reshape(activation[1], (64, 64))

		scaledGrid = grid * (255 / overallMaxValue)

		grids.append([activation[0], scaledGrid]);

		counter += 1

		if counter == 6*3:
			break

	fig, axes = plt.subplots(3, 6, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})

	fig.subplots_adjust(hspace=0.3, wspace=0.05)

	for ax, grid in zip(axes.flat, grids):
		ax.imshow(grid[1], 'Greys_r', interpolation='none')
		ax.set_title(grid[0])

	plt.show()
'''

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

def testKMeans(training, test, labelCount):
	clusters = akmp.multipleLabelsPerImage(training, labelCount, 500)

	filePath = "./clusters_small.npy"

	print 'Writing file ' + filePath
	if not os.path.exists(os.path.dirname(filePath)):
		os.makedirs(os.path.dirname(filePath))
	with open(filePath, "w") as outputFile:
		np.save(outputFile, clusters)

	clusters = akmp.clusterAnalysisMultipleLabels(clusters, training)

trainingInfo = './dumps/elastic_test/label_index_info_train.txt'
testInfo = './dumps/elastic_test/label_index_info_test.txt'
labelInfo = './dumps/elastic_test/label_index_mapping.txt'

trainingActivationVectorsFile = './training_vectors_elastic.npy'
testActivationVectorsFile = './test_vectors_elastic.npy'

batchSize = 100
batchLimit = 0

labelCount = 18
trainingActivationVectors = []
testActivationVectors = []

trainingJSONPath = ""
testJSONPath = ""

if(len(sys.argv) < 2):
	"No activation vectors provided."
else:
	trainingJSONPath = sys.argv[1]

if(len(sys.argv) < 3):
	"No activation vectors provided."
else:
	testJSONPath = sys.argv[2]


if trainingJSONPath.endswith('.npy'):
	trainingActivationVectors = anl.readVectorsFromJSON(trainingJSONPath)
else:
	trainingActivationVectors = anl.readDumpInfo(trainingInfo, batchSize, batchLimit, labelCount)
	anl.writeVectorsToJSON(trainingActivationVectors, trainingActivationVectorsFile)

if testJSONPath.endswith('.npy'):
	testActivationVectors = anl.readVectorsFromJSON(testJSONPath)
else:
	testActivationVectors = anl.readDumpInfo(testInfo, batchSize, batchLimit, labelCount)
	anl.writeVectorsToJSON(testActivationVectors, testActivationVectorsFile)

#drawVectors(trainingActivationVectors)

testKNN(trainingActivationVectors, testActivationVectors, labelCount)

# testKMeans(trainingActivationVectors, testActivationVectors, labelCount)
