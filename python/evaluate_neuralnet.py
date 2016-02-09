import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

root = './'

BATCH_SIZE = 5
MODEL_FILE = root + 'caffe_models/hybrid_cnn_handsorted/deploy.prototxt'
PRETRAINED_FILE = root + 'caffe_models/hybrid_cnn_handsorted/handsorted_iter_18000.caffemodel'
MEAN_FILE = root + 'image_imports/handsorted_lmdb/train_mean.binaryproto'
LAST_LAYER_NAME = 'fc8'

net = None
transformer = None

# TODO: Duplicate in predict_via_neuralnet.py: Move to module?

def setupCaffe():
   global net, transformer
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

def evaluateImageBatch(imagePaths):
   global net, transformer
   if net == None or transformer == None:
      logger.info('Initializing net and transformer.')
      setupCaffe()

   if len(imagePaths) == 0:
      logger.info("Found empty batch, skipping.")
      return []

   imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), imagePaths)

   net.blobs['data'].reshape(len(imagePaths), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
   net.blobs['data'].data[...] = imageData

   out = net.forward()
   result = np.zeros((1, out['prob'][0].shape[0]))

   counter = 0
   while counter < len(imagePaths):
      result[0,out['prob'][counter].argmax()] += 1
      counter +=1

   return result

if __name__ == '__main__':

   if len(sys.argv) != 4 and len(sys.argv) != 5:

      logger.info("Please provide as argument:")
      logger.info("1) Path to label_index_info_test.")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Target path for plot (*.pdf or *.png).")
      logger.info("4) Convusion matrix (*.npy).")
      sys.exit();

   labelIndexInfoPath = sys.argv[1]
   labelIndexMappingPath = sys.argv[2]
   plotFilePath = sys.argv[3]
   confusionMatrixPath = None

   if len(sys.argv) == 5:
      confusionMatrixPath = sys.argv[4]



   labels = []

   with open(labelIndexMappingPath, "r") as inputFile:
      for line in inputFile.readlines():
         labels.append(line.split(' ')[0])

   logger.info('Labels:')
   logger.info(labels)

   confusionMatrix = np.zeros((len(labels),len(labels)))
   if confusionMatrix == None:
      imagesByLabel = [[] for i in range(len(labels))]

      # split info by label
      with open(labelIndexInfoPath, "r") as inputFile:
         for line in inputFile.readlines():
            split = line.split(' ')
            imagePath = split[0]
            labelIndex = int(split[1])

            imagesByLabel[labelIndex].append(imagePath)

      # create batches per split if nessecary
      for labelIndex, imagePaths in enumerate(imagesByLabel):
         confusion = np.zeros((1,len(labels)))

         batches = []

         if len(imagePaths) > BATCH_SIZE:
            batchCounter = 0
            while batchCounter < len(imagePaths):
               batches.append(imagePaths[batchCounter:batchCounter + BATCH_SIZE])
               batchCounter += BATCH_SIZE
         else:
            batches.append(imagePaths)

         batchResults = np.empty((len(batches), len(labels)))
         # run net and collect predictions
         for batchIndex, batch in enumerate(batches):
            batchResults[batchIndex,:] = evaluateImageBatch(batch)
            # logger.debug(batchResults)
            # logger.debug(batchResults.shape)
         # sum up predictions per label type
         confusion = np.sum(batchResults, axis=0)
         # logger.debug(confusion)
         # logger.debug(confusion.shape)
         # logger.debug(np.sum(confusion))
         # append sums to matrix
         confusionMatrix[labelIndex,:] = confusion

      # logger.debug(confusionMatrix)

      confusionMatrixPath = 'confusion.npy'

      logger.info('Writing file ' + confusionMatrixPath)
      if not os.path.exists(os.path.dirname(confusionMatrixPath)) and os.path.dirname(confusionMatrixPath) != '':
         os.makedirs(os.path.dirname(confusionMatrixPath))
      with open(confusionMatrixPath, "w") as outputFile:
         np.save(outputFile, confusionMatrix)
   else:
      with open(confusionMatrixPath, 'r') as inputFile:
         confusionMatrix = np.load(inputFile)

   logger.debug(confusionMatrix.shape)
   maxValue = np.max(confusionMatrix, axis=1)
   logger.debug(maxValue.shape)
   logger.debug(maxValue)
   logger.debug(confusionMatrix[0,:])

   ax = plt.gca()
   scaled = (confusionMatrix.T / maxValue).T
   logger.debug(scaled[0,:])
   for(j,i),label in np.ndenumerate(confusionMatrix):
      ax.text(i,j, int(label), ha='center', va='center', color='green')


   plt.imshow(scaled, 'gray', interpolation='none')
   plt.show()
