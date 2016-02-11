import sys
import os
import logging
#os.environ['GLOG_minloglevel'] = '2' # comment to read full caffe log
import caffe
#os.environ['GLOG_minloglevel'] = '0'
import json
import numpy as np

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

batchSize = 200 # how many images to feed to to caffe as one batch
batchLimit = 0  # optional, how many batches should be processed (0 = until there are no more images)

root = './'

MODEL_FILE = root + 'caffe_models/hybrid_cnn_handsorted/deploy_FC7.prototxt'
PRETRAINED_FILE = root + 'caffe_models/hybrid_cnn_handsorted/hybridCNN_iter_700000.caffemodel'
MEAN_FILE = root + 'image_imports/handsorted_lmdb/train_mean.binaryproto'
LAST_LAYER_NAME = 'fc7'
USE_GPU = False

net = None
transformer = None

def setupCaffe():
   global net, transformer

   net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
   if USE_GPU:
      caffe.set_mode_gpu()
   else:
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

def crunchDumpFiles(dataInfoFilePath, batchSize, batchLimit, labelCount):
   imageBatchCount = 0
   imageCount = 0

   activations = []

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
            activations.extend(evaluateImageBatch(currentBatch, labelCount))

            currentBatchSize = 0
            currentBatch = []
            batchCount += 1

            logger.info(str(len(activations)))

            if batchCount == batchLimit and batchLimit != 0:
               break;

   activations.extend(evaluateImageBatch(currentBatch, labelCount))
   logger.info('Final number of images processed: ' + str(len(activations)))
   return activations


def evaluateImageBatch(imageBatch, labelCount):
   global net, transformer
   if net == None or transformer == None:
      setupCaffe()

   batchActivations = []
   imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x.get('path'))) , imageBatch)

   net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
   net.blobs['data'].data[...] = imageData

   out = net.forward()

   counter = 0
   for image in imageBatch:
      labelFlags = np.array([0] * labelCount)
      labelFlags[np.array(image.get('labelIds'), dtype=np.uint8)] += 1

      batchActivations.append(np.hstack((out[LAST_LAYER_NAME][counter], labelFlags)))

      counter += 1

   return batchActivations

def activationsToFile(activations, filePath):
   logger.info('Writing file ' + filePath)
   if not os.path.exists(os.path.dirname(filePath)) and os.path.dirname(filePath) != '':
      os.makedirs(os.path.dirname(filePath))
   with open(filePath, "w") as outputFile:
      np.save(outputFile, activations)

def calculateActivationVectors(labelIndexInfoPath, labelIndexMappingPath, resultPath):

      labelCount = 0
      with open(labelIndexMappingPath, 'r') as labelMappingFile:
         for line in labelMappingFile:
            labelCount += 1

      activationsToFile(crunchDumpFiles(labelIndexInfoPath, batchSize, batchLimit, labelCount), resultPath)

if __name__ == '__main__':
   if(len(sys.argv) != 4):
      logger.info("Please provide as arguments:")
      logger.info("1) path to label_index_info.txt")
      logger.info("2) path to label_index_mapping.txt")
      logger.info("3) target-filename for data (*.npy)")
      sys.exit()

   resultPath = sys.argv[3]

   if resultPath.endswith('/') == False:
      resultPath += '/'

   calculateActivationVectors(sys.argv[1], sys.argv[2], resultPath)
