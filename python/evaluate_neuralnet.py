import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import csv
import matplotlib.patches as mpatches

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

root = './'

BATCH_SIZE = 100
MODEL_FILE = root + 'caffe_models/hybrid_cnn_handsorted/deploy.prototxt'
PRETRAINED_FILE = root + 'caffe_models/hybrid_cnn_handsorted/handsorted_iter_18000.caffemodel'
MEAN_FILE = root + 'image_imports/handsorted_lmdb/train_mean.binaryproto'
LAST_LAYER_NAME = 'fc8'
USE_GPU = False

LOSS_MAXIMUM = 2
ITERATION_MAXIMUM = 1000

net = None
transformer = None

# TODO: Duplicate in predict_via_neuralnet.py: Move to module?

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

def createConfusionMatrix(labels, labelIndexInfoPath):
   imagesByLabel = [[] for i in range(len(labels))]
   confusionMatrix = np.zeros((len(labels),len(labels)))
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
      # sum up predictions per label type
      confusion = np.sum(batchResults, axis=0)
      # append sums to matrix
      confusionMatrix[labelIndex,:] = confusion

   return confusionMatrix

def plotConfusionMatrix(confusionMatrixPath, labels, evaluationTargetPath):
   maxValue = np.max(confusionMatrix, axis=1)

   ax = plt.gca()
   scaled = (confusionMatrix.T / maxValue).T

   for(j,i),label in np.ndenumerate(confusionMatrix):
      ax.text(i,j, int(label), ha='center', va='center', color='black',  fontsize=8)

   ticks = np.arange(len(labels))

   ax.set_yticks(ticks)
   ax.set_yticklabels(labels)
   ax.set_xticks(ticks)
   ax.set_xticklabels(labels, rotation=45, ha='right')

   # diagonal = np.diag(np.diag(scaled))
   # correct = np.ma.masked_array(scaled, mask=(diagonal==0))
   # false = np.ma.masked_array(scaled, mask=(diagonal!=0))
   # logger.debug(scaled)
   # logger.debug(diagonal)
   # logger.debug(correct)
   # logger.debug(correct)
   # logger.debug(false)
   #
   # logger.debug(false.shape)
   # logger.debug(scaled.shape)
   # logger.debug((correct == 0).shape)
   # pa = ax.imshow(correct, cmap='Greens', interpolation='none')
   # pb = ax.imshow(false, cmap='Reds', interpolation='none')

   plt.imshow(scaled, 'Blues', interpolation='none')
   plt.savefig(evaluationTargetPath + 'confusionMatrix.pdf', bbox_inches='tight')

def plotTrainingLossAndAccuracy(trainingLogCSVPath, testLogCSVPath, evaluationTargetPath):
   trainingLog = []
   with open(trainingLogCSVPath, 'r') as csvFile:
      reader = csv.reader(csvFile, delimiter=',')
      for row in reader:
         trainingLog.append(row)

   trainingLog = np.array(trainingLog)

   testLog = []
   with open(testLogCSVPath, 'r') as csvFile:
      reader = csv.reader(csvFile, delimiter=',')
      for row in reader:
         testLog.append(row)

   testLog = np.array(testLog)

   fig, ax1 = plt.subplots()
   ax1.plot(trainingLog[1:,0], trainingLog[1:,3], 'r', testLog[1:,0], testLog[1:,4], 'b')

   ax1.set_xlabel('Iterations')
   ax1.set_ylabel('Loss')

   ax2 = ax1.twinx()
   ax2.plot(testLog[1:,0], testLog[1:,3], 'g')
   ax2.set_ylabel('Accuracy')

   ax1.axis([0, ITERATION_MAXIMUM, 0, LOSS_MAXIMUM])
   ax2.axis([0, ITERATION_MAXIMUM, 0, 1])
   ax1.set_xticks(np.arange(0,ITERATION_MAXIMUM + 1,ITERATION_MAXIMUM / 10))
   ax1.set_xticklabels(np.arange(0,ITERATION_MAXIMUM + 1, ITERATION_MAXIMUM / 10), rotation=45)
   ax1.set_yticks(np.arange(0, LOSS_MAXIMUM, float(LOSS_MAXIMUM) / 10))
   ax2.set_yticks(np.arange(0, 1, float(1) / 10))

   ax1.grid(True)
   ax2.grid(True)

   labelTrainingLoss = mpatches.Patch(color='r', label='Training set loss')
   labelTestLoss = mpatches.Patch(color='b', label='Test set loss')
   labelTestAccuracy = mpatches.Patch(color='g', label='Test set accuracy')

   plt.legend(handles=[labelTrainingLoss, labelTestLoss, labelTestAccuracy])

   plt.savefig(evaluationTargetPath + 'lossAndAccuracy.pdf', bbox_inches='tight')
   plt.show()

if __name__ == '__main__':

   if len(sys.argv) != 6:

      logger.info("Please provide as argument:")
      logger.info("1) Path to label_index_info_test.")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Path to training log (*.csv).")
      logger.info("4) Path to test log (*.csv).")
      logger.info("5) Target path for evaluation results.")
      sys.exit();

   labelIndexInfoPath = sys.argv[1]
   labelIndexMappingPath = sys.argv[2]
   trainingLogCSVPath = sys.argv[3]
   testLogCSVPath = sys.argv[4]
   evaluationTargetPath = sys.argv[5]
   if evaluationTargetPath.endswith('/') == False:
      evaluationTargetPath += '/'

   if not os.path.exists(os.path.dirname(evaluationTargetPath)) and os.path.dirname(evaluationTargetPath) != '':
      os.makedirs(os.path.dirname(evaluationTargetPath))


   plotTrainingLossAndAccuracy(trainingLogCSVPath, testLogCSVPath, evaluationTargetPath)

   confusionMatrixPath = None

   if len(sys.argv) == 5:
      confusionMatrixPath = sys.argv[4]

   labels = []

   with open(labelIndexMappingPath, "r") as inputFile:
      for line in inputFile.readlines():
         labels.append(line.split(' ')[0])

   logger.info('Labels:')
   logger.info(labels)


   confusionMatrix = createConfusionMatrix(labels, labelIndexInfoPath)

   confusionMatrixPath = evaluationTargetPath + 'confusionMatrix.npy'
   logger.info('Writing file ' + confusionMatrixPath)
   with open(confusionMatrixPath, "w") as outputFile:
      np.save(outputFile, confusionMatrix)

   plotConfusionMatrix(confusionMatrix, labels, evaluationTargetPath)
