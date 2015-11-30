import sys
import os
import cv2
import Image
import random
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

import modules.arachne_caffe as anl
import modules.arachne_KNNPrediction as annp



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
		
	testNeighbours = annp.nearestNeighbours(training, test)

	count = 1
	data = []

	while count < len(training):
		(correct, wrong, correctPerLabel, wrongPerLabel) = annp.nNearestAnalysed(testNeighbours, count + 1, labelCount)
		data.append([count, float(correct)/(wrong + correct)])
		count += 1
		
	data = np.array(data)

	plt.plot(data[:,0], data[:,1], 'k')
	plt.axis([1, len(data), 0, 1])
	plt.grid(True)
	plt.show()


trainingInfo = './dumps/five_labels_small/label_index_info_train.txt'
testInfo = './dumps/five_labels_small/label_index_info_test.txt'
labelInfo = './dumps/five_labels_small/indexLabelMapping.txt'

trainingActivationVectorsFile = './trainingVectors.json'
testActivationVectorsFile = './testVectors.json'

batchSize = 100
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
	

if trainingJSONPath.endswith('.json'):
	(trainingActivationVectors) = anl.readVectorsFromJSON(trainingJSONPath)
else:
	trainingActivationVectors = anl.readDumpInfo(trainingInfo, batchSize, batchLimit) 
	anl.writeVectorsToJSON(trainingActivationVectors, trainingActivationVectorsFile)

if testJSONPath.endswith('.json'):
	testActivationVectors = anl.readVectorsFromJSON(testJSONPath)
else:
	testActivationVectors = anl.readDumpInfo(testInfo, batchSize, batchLimit)
	anl.writeVectorsToJSON(testActivationVectors, testActivationVectorsFile)

#testKNN(trainingActivationVectors, testActivationVectors, labelCount)
