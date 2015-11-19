import sys
import os
os.environ['GLOG_minloglevel'] = '2' 
import caffe
os.environ['GLOG_minloglevel'] = '0'
import cv2
import Image
import numpy as np
from scipy.misc import imresize
import json
import matplotlib.pyplot as plt

trainingInfo = './dumps/bauwerk_topo_buchseite_plastik_keramik_siegel/label_index_info_train.txt'
testInfo = './dumps/bauwerk_topo_buchseite_plastik_keramik_siegel/label_index_info_test.txt'
labelInfo = './dumps/bauwerk_topo_buchseite_plastik_keramik_siegel/indexLabelMapping.txt'

overallMaxValue = 0;
batchSize = 6 * 3

def evaluateImages(imageBatch):
		
	global overallMaxValue
	
	if len(imagePaths) == 0:
		print "No images found." 
		sys.exit()
		
	
	root = './'
	caffe_root = '/home/simon/Workspaces/caffe/'
	#MODEL_FILE = caffe_root + 'models/placesCNN/places205CNN_deploy.prototxt'
	#PRETRAINED = caffe_root + 'models/placesCNN/places205CNN_iter_300000.caffemodel'

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

	# Test self-made image

	#net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagePath))
	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x.get('path'))), imageBatch)
	
	net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData

	# out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
	out = net.forward()
	#print("Predicted class is #{}.".format(out['prob'][0].argmax()))

	# load labels
	#imagenet_labels_filename = caffe_root + 'models/placesCNN/categoryIndex_places205.csv'
	imagenet_labels_filename = root + 'examples/trained_models/custom_alex_shortened/categoryIndex_hybridCNN.csv'

	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\s')

	# sort top k predictions from softmax output
	

	counter = 0
	for image in imageBatch:
		
		#top_k = net.blobs['prob'].data[counter].flatten().argsort()[-1: -6: -1]
		#top_k_values = np.sort(out['prob'][counter].flatten())[-1: -6: -1]
		#np.set_printoptions(threshold=np.nan)
		#np.set_printoptions(suppress=True)		
		#print str(out['fc7'][counter])
		
		#print len(out['fc7'][counter])
		#print max(out['fc7'][counter])		
		
		methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
		
		maxValue = max(out['fc7'][counter])		
		if(maxValue > overallMaxValue):
			overallMaxValue = maxValue
			
		activationsVector.append([image.get('labelId'), out['fc7'][counter]])
		
		#grid =  np.reshape(out['fc7'][counter], (64, 64))
		
		#print 'Size: ', grid.size	
		#print 'Shape: ', grid.shape
		
		#scaledGrid = grid * (255 / maxValue)
		
		#grids.append([image.get('labelId'), scaledGrid]);
				
		#print top_k
		#print labels[top_k]
		
		#results.append([path.split('/')[-1], [labels[top_k].tolist(), top_k_values.tolist()]])
		
		counter += 1
		
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

imagePaths = [[]]

imageBatchCount = 0
imageCount = 0

activationsVector = []

if(len(sys.argv) != 2):
	print "Please provide image as parameter."
	sys.exit();
	
pathArgument = sys.argv[1]

with open(trainingInfo, "r") as result:
	currentBatch = []
	currentBatchSize = 0
	batchCount = 0
	for line in result.readlines():
		
		split = line.strip().split(' ')
		
		info = {'labelId':int(split[1]), 'path':split[0]}
		
		currentBatch.append(info)
		currentBatchSize += 1
		
		if currentBatchSize == batchSize:
			evaluateImages(currentBatch)
			currentBatchSize = 0
			currentBatch = []
			batchCount += 1
			print 'Activations collected: ' + str(len(activationsVector))
		
		if batchCount == 1:
			break


drawGrids(activationsVector)
