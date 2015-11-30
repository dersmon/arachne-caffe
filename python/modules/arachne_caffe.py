#os.environ['GLOG_minloglevel'] = '2' 
import caffe
#os.environ['GLOG_minloglevel'] = '0'
import json
import numpy as np

def getNetAndTransformer():
	
	root = './'
	caffe_root = '/home/simon/Workspaces/caffe/'
	
	MODEL_FILE = root + 'examples/trained_models/hybrid_cnn/hybridCNN_deploy_FC7.prototxt'
	PRETRAINED = root + 'examples/trained_models/hybrid_cnn/hybridCNN_iter_700000.caffemodel'

	net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
	caffe.set_mode_cpu()

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
	mean_file = np.array([104,117,123]) 
	transformer.set_mean('data', mean_file) #### subtract mean ####
	transformer.set_raw_scale('data', 255) # pixel value range
	transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR
	
	return (net, transformer)

def readDumpInfo(path, batchSize, batchLimit):	
	
	imageBatchCount = 0
	imageCount = 0
	
	activationVectors = []
		
	net, transformer = getNetAndTransformer()
	print '\nCollecting activations for ' + path + ':'
	
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
				
				activationVectors.extend(evaluateImageBatch(net, transformer, currentBatch))
				
				currentBatchSize = 0
				currentBatch = []
				batchCount += 1
								
				print str(len(activationVectors))
				if batchCount == batchLimit and batchLimit != 0:
					break;			
		
		activationVectors.extend(evaluateImageBatch(net, transformer, currentBatch))
					
	print str(len(activationVectors))
	
	return activationVectors

	
def evaluateImageBatch(net, transformer, imageBatch):
		
	batchActivations = []
	imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x.get('path'))) , imageBatch)
	
	net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
	net.blobs['data'].data[...] = imageData
	
	out = net.forward()
	
	counter = 0
	for image in imageBatch:			
		batchActivations.append(np.hstack((image.get('labelId'), out['fc7'][counter])))		
		counter += 1
		
	return batchActivations

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
