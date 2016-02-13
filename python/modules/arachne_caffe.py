import sys
import os
#os.environ['GLOG_minloglevel'] = '2' # comment to read full caffe log
import caffe
#os.environ['GLOG_minloglevel'] = '0'
import json
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

root = './'

MODEL_FILE = root + 'caffe_models/hybrid_cnn_handsorted/deploy_FC7.prototxt'
PRETRAINED_FILE = root + 'caffe_models/hybrid_cnn_handsorted/handsorted_iter_18000.caffemodel'
MEAN_FILE = root + 'image_imports/handsorted_lmdb/train_mean.binaryproto'
LAST_LAYER_NAME = 'fc7'


def getNetAndTransformer():

	net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
	caffe.set_mode_cpu()

	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(MEAN_FILE).read()
	blob.ParseFromString(data)
	mean = np.array(caffe.io.blobproto_to_array(blob))[0].mean(1).mean(1)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)
	transformer.set_mean('data', mean)
	transformer.set_channel_swap('data', (2,1,0))

	return (net, transformer)

def crunchDumpFiles(dataInfoFilePath, batchSize, batchLimit, labelCount):

	imageBatchCount = 0
	imageCount = 0

	activations = []

	net, transformer = getNetAndTransformer()
	logger.info('Calculating activations for images mentioned in ' + dataInfoFilePath + ':')

	with open(dataInfoFilePath, "r") as result:
		currentBatch = []
		currentBatchSize = 0
		batchCount = 0
		for line in result.readlines():

			split = line.strip().split(' ')

			info = {'path':split[0], 'labelIds':split[1:]}

			currentBatch.append(info)
			currentBatchSize += 1

			if currentBatchSize == batchSize:
				activations.extend(evaluateImageBatch(net, transformer, currentBatch, labelCount))

				currentBatchSize = 0
				currentBatch = []
				batchCount += 1

				logger.info(str(len(activations)))

				if batchCount == batchLimit and batchLimit != 0:
					break;

	activations.extend(evaluateImageBatch(net, transformer, currentBatch, labelCount))
	logger.info('Final number of images processed: ' + str(len(activations)))
	return activations


def evaluateImageBatch(net, transformer, imageBatch, labelCount):

	batchActivations = []
	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x.get('path'))) , imageBatch)

	net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData

	out = net.forward()

	counter = 0
	for image in imageBatch:
		if labelCount != 0:
			labelFlags = np.array([0] * labelCount)
			labelFlags[np.array(image.get('labelIds'), dtype=np.uint8)] += 1

			batchActivations.append(np.hstack((out[LAST_LAYER_NAME][counter], labelFlags)))
		else:
			batchActivations.append(out[LAST_LAYER_NAME][counter])

		counter += 1

	return batchActivations

def activationsToFile(activations, filePath):
	logger.info('Writing file ' + filePath)
	if not os.path.exists(os.path.dirname(filePath)) and os.path.dirname(filePath) != '':
		os.makedirs(os.path.dirname(filePath))
	with open(filePath, "w") as outputFile:
		np.save(outputFile, activations)

def activationsFromFile(filePath):
	logger.info('Opening file: ' + filePath)
	with open(filePath, 'r') as inputFile:
		return np.load(inputFile)
