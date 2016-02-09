import sys
import caffe
import cv2
import Image
import os
import numpy as np
from scipy.misc import imresize
import json
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

batchSize = 10

def evaluateImageBatch(imagePaths):

	if len(imagePaths) == 0:
		logger.info("Found empty batch, skipping.")
		return []

	root = './'
	caffe_root = '/home/simon/Workspaces/caffe/'
	#MODEL_FILE = caffe_root + 'models/placesCNN/places205CNN_deploy.prototxt'
	#PRETRAINED = caffe_root + 'models/placesCNN/places205CNN_iter_300000.caffemodel'

	MODEL_FILE = root + 'examples/trained_models/hybrid_cnn_handsorted/hybridCNN_deploy.prototxt'
	PRETRAINED = root + 'examples/trained_models/hybrid_cnn_handsorted/hybridCNN_handsorted_iter_18000.caffemodel'

	net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
	caffe.set_mode_cpu()

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
	mean_file = np.array([104,117,123])
	transformer.set_mean('data', mean_file) #### subtract mean ####
	transformer.set_raw_scale('data', 255) # pixel value range
	transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(root + 'examples/trained_models/hybrid_cnn/hybridCNN_mean.binaryproto' , 'rb' ).read()
	blob.ParseFromString(data)

	meanArray = np.array( caffe.io.blobproto_to_array(blob) ).transpose(3,2,1,0)
	meanArray = meanArray[:,:,:,0]

	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), imagePaths)

	net.blobs['data'].reshape(len(imagePaths),net.blobs['data'].shape[1],net.blobs['data'].shape[2],net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData

	# out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
	out = net.forward()
	# logger.info("Predicted class is #{}.".format(out['prob'][0].argmax()))

	# load labels
	imagenet_labels_filename = root + 'image_dumps/handsorted/label_index_mapping_0.txt'

	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\s')

	result = []

	counter = 0
	for path in imagePaths:

		top_k = net.blobs['prob'].data[counter].flatten().argsort()[-1: -6: -1]
		top_k_values = np.sort(out['prob'][counter].flatten())[-1: -6: -1]

		result.append([os.path.basename(path), [labels[top_k].tolist(), top_k_values.tolist()]])

		counter += 1

	return result

if __name__ == '__main__':
	if(len(sys.argv) != 3):
		logger.info("Please provide as argument:")
		logger.info("1) Image or folder containing images.")
		logger.info("2) Target path for evaluation results (*.json).")
		sys.exit();

	pathArgument = sys.argv[1]
	resultPath = sys.argv[2]

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

	result = []
	for batch in imagePaths:
		result.extend(evaluateImageBatch(batch))

	resultJSON = json.dumps(result)

	logger.info("Writing results to " + resultPath + ".")

	if not os.path.exists(os.path.dirname(resultPath)) and os.path.dirname(resultPath) != "":
		os.makedirs(os.path.dirname(resultPath))

	with open(resultPath, "w") as result:
		result.write(resultJSON)
