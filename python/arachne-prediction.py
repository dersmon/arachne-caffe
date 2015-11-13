import sys
import caffe
import cv2
import Image
import os
import numpy as np
from scipy.misc import imresize
import json

batchSize = 10
results = []
resultPath = "results.json"

def retreiveImagePaths():
	print "Todo"
	#Todo

def evaluateImage(imagePaths):
		
	if len(imagePaths) == 0:
		print "No images found." 
		sys.exit()
		
	root = './'
	caffe_root = '/home/simon/Workspaces/caffe/'
	#MODEL_FILE = caffe_root + 'models/placesCNN/places205CNN_deploy.prototxt'
	#PRETRAINED = caffe_root + 'models/placesCNN/places205CNN_iter_300000.caffemodel'

	MODEL_FILE = root + 'examples/trained_models/custom_alex/deploy.prototxt'
	PRETRAINED = root + 'examples/trained_models/custom_alex/caffe_alexnet_train_iter_1500.caffemodel'

	net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
	caffe.set_mode_cpu()

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
	mean_file = np.array([104,117,123]) 
	transformer.set_mean('data', mean_file) #### subtract mean ####
	transformer.set_raw_scale('data', 255) # pixel value range
	transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR


	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(root + 'examples/trained_models/custom_alex/mean_train.binaryproto' , 'rb' ).read()
	blob.ParseFromString(data)

	meanArray = np.array( caffe.io.blobproto_to_array(blob) ).transpose(3,2,1,0)
	meanArray = meanArray[:,:,:,0]

	# Test self-made image

	#net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagePath))
	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), imagePaths)
	
	net.blobs['data'].reshape(len(imagePaths),net.blobs['data'].shape[1],net.blobs['data'].shape[2],net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData

	# out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
	out = net.forward()
	print("Predicted class is #{}.".format(out['prob'][0].argmax()))

	# load labels
	#imagenet_labels_filename = caffe_root + 'models/placesCNN/categoryIndex_places205.csv'
	imagenet_labels_filename = root + 'examples/trained_models/custom_alex/indexLabelMapping.txt'

	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\s')

	# sort top k predictions from softmax output
	

	counter = 0
	for path in imagePaths:
		
		top_k = net.blobs['prob'].data[counter].flatten().argsort()[-1: -6: -1]
		top_k_values = np.sort(out['prob'][counter].flatten())[-1: -6: -1]
		#print len(out['prob'])
		#print out['prob'][counter]
		#print top_k
		#print labels[top_k]
		
		results.append([path, [labels[top_k].tolist(), top_k_values.tolist()]])
		
		counter += 1
		

if(len(sys.argv) != 2):
	print "Please provide image as parameter."
	sys.exit();
	
pathArgument = sys.argv[1]

imagePaths = [[]]

imageBatchCount = 0
imageCount = 0

if os.path.isfile(pathArgument):
	imagePaths[0].append(pathArgument)
else: 
	for fn in os.listdir(pathArgument):		
		if os.path.isfile(pathArgument + "/" + fn):			
			imagePaths[imageBatchCount].append(pathArgument + "/" + fn)
			imageCount += 1
			
			if(imageCount % batchSize == 0):
				imageBatchCount += 1
				imagePaths.append([])

for pathBatch in imagePaths:
	evaluateImage(pathBatch)

print str(results)

resultJSON = json.dumps(results)

if not os.path.exists(os.path.dirname(resultPath)):
	os.makedirs(os.path.dirname(resultPath))

with open(resultPath, "a") as result:
	result.write(resultJSON)	



