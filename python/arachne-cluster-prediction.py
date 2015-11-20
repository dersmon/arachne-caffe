import sys
import os
os.environ['GLOG_minloglevel'] = '2' 
import caffe
os.environ['GLOG_minloglevel'] = '0'
import cv2
import Image
import numpy as np
import random
from scipy.misc import imresize
import json
import matplotlib.pyplot as plt

trainingInfo = './dumps/bauwerk_topo_buchseite_plastik_keramik_siegel/label_index_info_train.txt'
testInfo = './dumps/bauwerk_topo_buchseite_plastik_keramik_siegel/label_index_info_test.txt'
labelInfo = './dumps/bauwerk_topo_buchseite_plastik_keramik_siegel/indexLabelMapping.txt'

trainingActivationVectorsFile = './trainingVectors.json'
testActivationVectorsFile = './testVectors.json'

overallMaxValue = 0;

batchSize = 100
batchLimit = 0

labelCount = 6
trainingActivationVectors = []
testActivationVectors = []

def readDumpInfo(path):	
	
	global batchSize, batchLimit
	
	imageBatchCount = 0
	imageCount = 0
	
	activationVectors = []
	
	with open(path, "r") as result:
		currentBatch = []
		currentBatchSize = 0
		batchCount = 0
		for line in result.readlines():
			
			split = line.strip().split(' ')
			
			info = {'labelId':int(split[1]), 'path':split[0]}
			
			currentBatch.append(info)
			currentBatchSize += 1
			
			if currentBatchSize == batchSize:
				activationVectors.extend(evaluateImages(currentBatch))
				currentBatchSize = 0
				currentBatch = []
				batchCount += 1
								
				print 'Activations collected: ' + str(len(activationVectors))
					
				if batchCount == batchLimit and batchLimit != 0:
					break;			
	
	return activationVectors
		
def evaluateImages(imageBatch):
		
	global overallMaxValue
	
	if len(imageBatch) == 0:
		print "No images found." 
		sys.exit()
		
	batchActivations = []
	
	root = './'
	caffe_root = '/home/simon/Workspaces/caffe/'
	
	MODEL_FILE = root + 'examples/trained_models/custom_alex_shortened/hybridCNN_deploy_FC7.prototxt'
	PRETRAINED = root + 'examples/trained_models/custom_alex_shortened/hybridCNN_iter_700000.caffemodel'

	net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
	caffe.set_mode_cpu()

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
	mean_file = np.array([104,117,123]) 
	transformer.set_mean('data', mean_file) #### subtract mean ####
	transformer.set_raw_scale('data', 255) # pixel value range
	transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR


	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(root + 'examples/trained_models/custom_alex_shortened/hybridCNN_mean.binaryproto' , 'rb' ).read()
	blob.ParseFromString(data)

	meanArray = np.array( caffe.io.blobproto_to_array(blob) ).transpose(3,2,1,0)
	meanArray = meanArray[:,:,:,0]

	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x.get('path'))), imageBatch)
	
	net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData
	
	out = net.forward()

	counter = 0
	for image in imageBatch:		
		
		maxValue = max(out['fc7'][counter])		
		if maxValue > overallMaxValue:
			overallMaxValue = maxValue
			
		batchActivations.append([image.get('labelId'), out['fc7'][counter]])
		
		counter += 1
		
	return batchActivations
		
def kMeans(activationVectors):
	global labelCount
	
	centers = []
	count = 0
	running = 1
	while count < labelCount:
		centers.append({'position':random.choice(activationVectors)[1], 'clusterMembers': []})
		count += 1	
			
	kMeansIteration(centers, activationVectors)
	
	count = 0
	
	while count < 100:
		centers = kMeansIteration(centers, activationVectors)
		count += 1
		centerCounter = 0
		#print '\n'
		for center in centers:
			#print 'Cluster ' + str(centerCounter + 1) + ', labels:'
			points = [activationVectors[i] for i in center['clusterMembers']]
			labels = [0, 0, 0, 0, 0, 0]
			
			for point in points:
				labels[point[0]] += 1			
			
			#print str(labels)
			centerCounter += 1
	
	return centers

def kMeansIteration(centers, activations):
	
	tempCenters = []
	updatedCenters = []
	
	for center in centers:
		tempCenters.append({'position':center.get('position'), 'clusterMembers': []})	
		
	count = 0
	for activation in activations:
		
		distances = []			
		
		# Calculate distance to centers
		for center in centers:
			difference = center['position'] - activation[1]
			distances.append(np.linalg.norm(difference))
			
		# Assign to closest center		
		members = tempCenters[np.argmin(distances)].get('clusterMembers')
		members.append(count)		
		tempCenters[np.argmin(distances)]['clusterMembers'] = members
		
		count += 1
	
	
	# Adjust centers towards mean of assigned vectors
	
	for center in tempCenters:
		#print "Assigned points: " + str(center['clusterMembers'])
		points = [activations[i][1] for i in center['clusterMembers']]	
	
		updatedPosition = np.sum(points, axis=0)
		updatedPosition = updatedPosition / len(center['clusterMembers'])
		#print 'old position: ' + str(center['position']) + ', length: ' + str(len(center['position']))
		#print 'new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0])
		
		updatedCenters.append({'position':updatedPosition, 'clusterMembers':center['clusterMembers']})
	
	return updatedCenters	
	

def writeVectorsToJSON(activationVectors, filePath):
	
	activationVectorsAsList = [];

	for vector in activationVectors:	
		activationVectorsAsList.append([vector[0], vector[1].tolist()])
		
	resultJSON = json.dumps(activationVectorsAsList)

	if not os.path.exists(os.path.dirname(filePath)):
		os.makedirs(os.path.dirname(filePath))

	with open(filePath, "a") as outputFile:
		outputFile.write(resultJSON)	

def readVectorsFromJSON(filePath):
	
	with open(filePath) as fileData:
		data = json.load(fileData)
	
	activationVectors = []
	
	for vector in data:	
		activationVectors.append([vector[0], np.array(vector[1])])
		
	return activationVectors

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
	trainingActivationVectors = readVectorsFromJSON(trainingJSONPath)
else:
	trainingActivationVectors = readDumpInfo(trainingInfo)
	writeVectorsToJSON(trainingActivationVectors, trainingActivationVectorsFile)

if testJSONPath.endswith('.json'):
	testActivationVectors = readVectorsFromJSON(testJSONPath)
else:
	testActivationVectors = readDumpInfo(testInfo)
	writeVectorsToJSON(testActivationVectors, testActivationVectorsFile)


clusterCenters = kMeans(trainingActivationVectors)

labelPerCluster = []

clusterCounter = 0

for cluster in clusterCenters:
	#print 'Cluster ' + str(clusterCounter + 1) + ', labels:'
	points = [trainingActivationVectors[i] for i in cluster['clusterMembers']]
	labels = [0, 0, 0, 0, 0, 0]
		
	for point in points:
		labels[point[0]] += 1			
			
	#print str(labels)
	
	labelPerCluster.append({'clusterId': clusterCounter, 'position': cluster['position'], 'label': np.argmax(labels)})
	clusterCounter += 1






tempCenters = []
updatedCenters = []

correct = 0
wrong = 0

correctPerLabel = [0, 0, 0, 0, 0, 0]
wrongPerLabel = [0, 0, 0, 0, 0, 0]

for activation in testActivationVectors:
	
	distances = []			
	
	# Calculate distance to centers
	for center in clusterCenters:
		difference = center['position'] - activation[1]
		distances.append(np.linalg.norm(difference))
		
	for center in labelPerCluster:
		if center['clusterId'] == np.argmin(distances):
			#print 'Assigned cluster ' + str(center['clusterId']) + ' with label ' + str(center['label']) + ' to image with label ' + str(activation[0])
			if center['label'] == activation[0]:
				correct += 1
				correctPerLabel[activation[0]] += 1
			else:
				wrong += 1				
				wrongPerLabel[activation[0]] += 1

print 'correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(wrong/correct)
print 'correct per label: ' + str(correctPerLabel)
print 'wrong per label: 'str(wrongPerLabel)
#drawGrids(activationsVector)
