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
	labels = np.array(vectors)[:,0]
	grid = np.array(vectors)[:,1:]
		
	maxValue = np.amax(grid)	
	minValue = np.amin(grid)
	
	# print 'max activation: ' + str(maxValue) + ', min activation: ' + str(minValue)
	
	scaled = grid * (255 / maxValue)
	
	scaled = scaled.reshape(scaled.shape[0],4,1024)
	scaled = scaled.mean(axis=1)
	
	plt.imshow(scaled, 'Greys_r', interpolation='none')
	plt.show()

def testKNN(training, test, labelCount):
		
	testNeighbours = aknnp.nearestNeighbours(training, test)

	count = 0
	data = []

	while count < len(training):
		print 'Calculating prediction for ' + str(count + 1) + ' nearest neighbours.' 
		(correct, wrong, correctPerLabel, wrongPerLabel) = aknnp.nNearestAnalysed(testNeighbours, count + 1, labelCount)
		data.append([count, float(correct)/(wrong + correct)])
		count += 1
		
	data = np.array(data)

	plt.plot(data[:,0], data[:,1], 'k')
	plt.axis([1, len(data), 0, 1])
	plt.grid(True)
	plt.show()

def testKMeans(training, test, labelCount):
	(splitTraining, splitClusters) = akmp.multipleClusterPerLabe(training, labelCount, 3)
	count = 0
	print 'training: ' + str(len(splitTraining)) + ', clusters:' + str(len(splitClusters))
	print splitTraining[0][0]
	print splitTraining[0][0][0]
	while(count < labelCount):
		#akmp.clusterAnalysis(splitTraining[count], splitClusters[count], labelCount)
		count += 1
	
	#clusters = akmp.kMeans(training, labelCount)
	#clusters = akmp.clusterAnalysis(training, clusters, labelCount)
	#akmp.clusterTest(clusters, test, labelCount)
	
trainingInfo = './dumps/five_labels/label_index_info_train.txt'
testInfo = './dumps/five_labels/label_index_info_test.txt'
labelInfo = './dumps/five_labels/indexLabelMapping.txt'

trainingActivationVectorsFile = './trainingVectors.npy'
testActivationVectorsFile = './testVectors.npy'

batchSize = 300
batchLimit = 0

labelCount = 5
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
	trainingActivationVectors = anl.readDumpInfo(trainingInfo, batchSize, batchLimit) 
	anl.writeVectorsToJSON(trainingActivationVectors, trainingActivationVectorsFile)

if testJSONPath.endswith('.npy'):
	testActivationVectors = anl.readVectorsFromJSON(testJSONPath)
else:
	testActivationVectors = anl.readDumpInfo(testInfo, batchSize, batchLimit)
	anl.writeVectorsToJSON(testActivationVectors, testActivationVectorsFile)

testKNN(trainingActivationVectors, testActivationVectors, labelCount)

#testKMeans(trainingActivationVectors, testActivationVectors, labelCount)
