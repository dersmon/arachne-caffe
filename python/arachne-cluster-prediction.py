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

trainingInfo = './dumps/five_labels_small/label_index_info_train.txt'
testInfo = './dumps/five_labels_small/label_index_info_test.txt'
labelInfo = './dumps/five_labels_small/indexLabelMapping.txt'

trainingActivationVectorsFile = './trainingVectors.json'
testActivationVectorsFile = './testVectors.json'

overallMaxValue = 0;

batchSize = 100
batchLimit = 0

labelCount = 6
trainingActivationVectors = []
testActivationVectors = []

def getNetAndTransformer():
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
	
	return (net, transformer)

def readDumpInfo(path):	
	
	global batchSize, batchLimit
	
	imageBatchCount = 0
	imageCount = 0
	
	activationVectors = []
	
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
				
				activationVectors.extend(evaluateImages(net, transformer, currentBatch))
				
				currentBatchSize = 0
				currentBatch = []
				batchCount += 1
								
				print 'Activations collected: ' + str(len(activationVectors))
				if batchCount == batchLimit and batchLimit != 0:
					break;			
		
		activationVectors.extend(evaluateImages(net, transformer, currentBatch))
				
					
	print 'Activations collected: ' + str(len(activationVectors))
	return activationVectors
		
def evaluateImages(net, transformer, imageBatch):
		
	global overallMaxValue

	print 'batch size: ' + str(len(imageBatch))
	#blob = caffe.proto.caffe_pb2.BlobProto()
	#data = open(root + 'examples/trained_models/custom_alex_shortened/hybridCNN_mean.binaryproto' , 'rb' ).read()
	#blob.ParseFromString(data)

	#meanArray = np.array( caffe.io.blobproto_to_array(blob) ).transpose(3,2,1,0)
	#meanArray = meanArray[:,:,:,0]

	#np.vstack((a,b))
	batchActivations = []
	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x.get('path'))) , imageBatch)
	
	net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData
	
	out = net.forward()
	
	counter = 0
	for image in imageBatch:		
		
		maxValue = max(out['fc7'][counter])		
		if maxValue > overallMaxValue:
			overallMaxValue = maxValue
		
		batchActivations.append(np.hstack((image.get('labelId'), out['fc7'][counter])))		
		counter += 1
		
	# print str(batchActivations[:,0]) # prints label row
	
	return batchActivations
		
def kMeans(activationVectors):
	global labelCount
	
	centers = []
	count = 0
	running = 1
	while count < labelCount:
		centers.append({'position':random.choice(activationVectors)[1:], 'clusterMembers': []})
		count += 1	
			
	count = 0
	while count < 50:
		centers = kMeansIteration(centers, activationVectors)
		count += 1
		centerCounter = 0
		print '\n'
		for center in centers:
			print 'Cluster ' + str(centerCounter + 1) + ', labels:'
			points = [activationVectors[i] for i in center['clusterMembers']]
			labels = [0, 0, 0, 0, 0, 0]
			
			for point in points:
				labels[int(point[0])] += 1			
			
			print str(labels)
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
			difference = center['position'] - activation[1:]
			distances.append(np.linalg.norm(difference))
			
		# Assign to closest center		
		members = tempCenters[np.argmin(distances)].get('clusterMembers')
		members.append(count)		
		tempCenters[np.argmin(distances)]['clusterMembers'] = members
		
		count += 1
	
	
	# Adjust centers towards mean of assigned vectors
	
	for center in tempCenters:
		#print "Assigned points: " + str(center['clusterMembers'])
		points = [activations[i][1:] for i in center['clusterMembers']]	
	
		updatedPosition = np.sum(points, axis=0)
		if len(center['clusterMembers']) != 0:
			updatedPosition /= len(center['clusterMembers'])
		
		#print 'old position: ' + str(center['position']) + ', length: ' + str(len(center['position']))
		#print 'new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0])
		
		updatedCenters.append({'position':updatedPosition, 'clusterMembers':center['clusterMembers']})
	
	return updatedCenters	
	

def writeVectorsToJSON(activationVectors, filePath):
	
	activationVectorsAsList = [];
	if not os.path.exists(os.path.dirname(filePath)):				
		os.makedirs(os.path.dirname(filePath))
	with open(filePath, "a") as outputFile:
		print 'Writing to file ' + str(filePath)
		for vector in activationVectors:	
			activationVectorsAsList.append(vector.tolist())
	
		resultJSON = json.dumps(activationVectorsAsList)
		
		outputFile.write(resultJSON)			

def readVectorsFromJSON(filePath):
	
	with open(filePath, 'r') as fileData:
		data = json.load(fileData)
	
	activationVectors = []
	
	for vector in data:	
		activationVectors.append(np.array(vector))
		
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
	print 'Cluster ' + str(clusterCounter + 1) + ', labels:'
	points = [trainingActivationVectors[i] for i in cluster['clusterMembers']]
	labels = [0, 0, 0, 0, 0, 0]
		
	for point in points:
		labels[int(point[0])] += 1			
			
	print str(labels)
	
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
		difference = center['position'] - activation[1:]
		distances.append(np.linalg.norm(difference))
		
	for center in labelPerCluster:
		if center['clusterId'] == np.argmin(distances):
			#print 'Assigned cluster ' + str(center['clusterId']) + ' with label ' + str(center['label']) + ' to image with label ' + str(activation[0])
			if center['label'] == int(activation[0]):
				correct += 1
				correctPerLabel[int(activation[0])] += 1
			else:
				wrong += 1				
				wrongPerLabel[int(activation[0])] += 1

print 'correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(float(correct)/(wrong + correct))
print 'correct per label: ' + str(correctPerLabel)
print 'wrong per label: ' + str(wrongPerLabel)

net, transformer = getNetAndTransformer()

testImage = {'labelId':0, 'path':'/home/simon/Workspaces/python/arachne-caffe/examples/images/Statue_of_Liberty_Paris_001.jpg'}

testVector = evaluateImages(net, transformer, [testImage])[0]
print testVector.shape
distances = []			
	
# Calculate distance to centers
for center in clusterCenters:
	difference = center['position'] - testVector[1:]
	distances.append(np.linalg.norm(difference))

print str(distances)
print str(labelPerCluster)
	
for center in labelPerCluster:
	if center['clusterId'] == np.argmin(distances):
		print 'Assigned cluster ' + str(center['clusterId']) + ' with label ' + str(center['label']) + ' to image with label ' + str(testVector[0])
			

		
##drawGrids(activationsVector)
