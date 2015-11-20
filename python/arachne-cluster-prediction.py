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

trainingInfo = './dumps/six_labels_small/label_index_info_train.txt'
testInfo = './dumps/six_labels_small/label_index_info_test.txt'
labelInfo = './dumps/six_labels_small/indexLabelMapping.txt'

trainingActivationVectorsFile = './trainingVectors.json'

overallMaxValue = 0;

batchSize = 100
batchLimit = 0

labelCount = 6
trainingActivationVectors = []

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
	running = true
	while count < labelCount:
		centers.append(random.choice(activationVectors)[1])
		count += 1
	
	for activationVector in activationVectors:
		
		distances = []			
		
		# Calculate distance to centers
		for center in centers:
			difference = center - activationVector[1]
			distances.append(np.linalg.norm(difference))
			
		# Assign to closest center
		print str(distances)
	# Adjust centers towards center of assigned vectors
	
		
	
	print 'Todo'

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

pathArgument = ""

if(len(sys.argv) < 2):
	"No activation vectors provided."
else:
	pathArgument = sys.argv[1]
	
if pathArgument.endswith('.json'):
	readVectorsFromJSON(pathArgument)
else:
	trainingActivationVectors = readDumpInfo(trainingInfo)
	writeVectorsToJSON(trainingActivationVectors, trainingActivationVectorsFile)

#drawGrids(activationsVector)
