import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import csv
import matplotlib.patches as mpatches
import modules.utility as utility

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

root = './'
caffe_root = '/home/simon/Development/caffe'

sys.path.append(caffe_root + '/tools/extra/')
import parse_log as pl


BATCH_SIZE = 100
MODEL_FILE = root + 'caffe_models/hybrid_cnn_handsorted_lmdb/deploy.prototxt'
# PRETRAINED_FILE = root + 'caffe_models/hybrid_cnn_handsorted/handsorted_iter_18000.caffemodel'
PRETRAINED_FILE = root + 'caffe_models/hybrid_cnn_handsorted_lmdb/snapshots/convolutional_001_8000_iter_20000.caffemodel'
MEAN_FILE = root + 'image_imports/handsorted_lmdb/mean_test.binaryproto'
USE_GPU = False

LOSS_MAXIMUM = 5
ITERATION_MAXIMUM = 20000
ITERATION_VARIATIONS = 3

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
   return out['prob']

def testNeuralNet(labels, labelIndexInfoPath, targetFolder):
   imagesByLabel = [[] for i in range(len(labels))]
   confusionMatrix = np.zeros((len(labels),len(labels)))
   # split info by label


   with open(labelIndexInfoPath, "r") as inputFile:
      for line in inputFile.readlines():
         split = line.split(' ')
         imagePath = split[0]
         labelIndex = int(split[1])

         imagesByLabel[labelIndex].append(imagePath)

   overallCorrect = 0
   overallWrong = 0
   meanAveragePrecision = 0

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
         batchResult = evaluateImageBatch(batch)
         # logger.debug(batchResult)
         # logger.debug(batchResult.shape)

         predictions = np.argmax(batchResult, axis=1)
         # logger.debug(predictions.shape)
         # logger.debug(predictions)

         predictionCounter = 0
         while predictionCounter < predictions.shape[0]:
            confusion[0,predictions[predictionCounter]] += 1

            if predictions[predictionCounter] == labelIndex:
               overallCorrect += 1
            else:
               overallWrong += 1

            # logger.debug(confusion)
            # logger.debug(overallCorrect)
            # logger.debug(overallWrong)

            predictionCounter += 1

         resultCounter = 0
         while resultCounter < batchResult.shape[0]:
            sortedPredictedLabel = np.argsort(batchResult[resultCounter])[::-1].tolist()

            averagePrecision = 0
            relevant = 0

            for idx, value in enumerate(sortedPredictedLabel):
               indicator = 0
               if(value == labelIndex):
                  indicator = 1
                  relevant += 1

               precision = float(relevant) / (idx + 1)
               averagePrecision += (precision * indicator)

            if relevant != 0:
               averagePrecision = float(averagePrecision) / relevant
            meanAveragePrecision += averagePrecision

            resultCounter += 1

      confusionMatrix[labelIndex,:] = confusion

   logger.info('Writing file ' + targetFolder + "overview.csv")
   np.savetxt( targetFolder + "overview.csv", np.array([meanAveragePrecision, overallCorrect, overallWrong, float(overallCorrect) / (overallCorrect + overallWrong)]), delimiter=',')


   logger.info(' Accuracy: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))
   meanAveragePrecision = float(meanAveragePrecision) / (overallWrong + overallCorrect)
   logger.info(' Mean average precision: '+str(meanAveragePrecision))

   return [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong]

def plotTrainingLossAndAccuracy(trainingLogPath, evaluationTargetPath):

   trainingLog, testLog = pl.parse_log(trainingLogPath)
   # logger.debug(testLog[1])

   trainingData = []
   for item in trainingLog:
      trainingData.append([item['NumIters'], item['loss']])
   testData = []
   for item in testLog:
      testData.append([item['NumIters'], item['loss'], item['accuracy']])

   trainingData = np.array(trainingData)
   testData = np.array(testData)


   trainingLog = np.array(trainingLog)
   testLog = np.array(testLog)

   # logger.debug(trainingLog.shape)
   # logger.debug(testLog.shape)


   iterationMaximum = ITERATION_MAXIMUM
   counter = 0
   while counter < ITERATION_VARIATIONS:

      fig, ax1 = plt.subplots()
      trainingLossPlot, = ax1.plot(trainingData[:,0], trainingData[:,1], color='r', label='Training set loss')
      testLossPlot, = ax1.plot(testData[:,0], testData[:,1], label='Test set loss', color='b')

      ax1.set_xlabel('Iterations')
      ax1.set_ylabel('Loss')

      ax2 = ax1.twinx()
      accuracyPlot, = ax2.plot(testData[:,0], testData[:,2], label='Test set accuracy', color='g')
      ax2.set_ylabel('Accuracy')

      ax1.axis([0, iterationMaximum, 0, LOSS_MAXIMUM])
      ax2.axis([0, iterationMaximum, 0, 1])
      ax1.set_xticks(np.arange(0,iterationMaximum + 1, iterationMaximum * 0.1))
      ax1.set_xticklabels(np.arange(0,iterationMaximum + 1, iterationMaximum  * 0.1), rotation=45)
      ax1.set_yticks(np.arange(0, LOSS_MAXIMUM, float(LOSS_MAXIMUM) / 10))
      ax2.set_yticks(np.arange(0, 1, float(1) / 10))

      ax1.grid(True)
      ax2.grid(True)

      # plt.title(evaluationTargetPath)

      plt.legend([trainingLossPlot, testLossPlot, accuracyPlot], [trainingLossPlot.get_label(), testLossPlot.get_label(), accuracyPlot.get_label()], bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)

      plt.savefig(evaluationTargetPath +'lossAndAccuracy_' + str(iterationMaximum) + '.pdf', bbox_inches='tight')

      plt.close()

      iterationMaximum = int(iterationMaximum * 0.5)

      counter += 1

if __name__ == '__main__':

   if len(sys.argv) != 5:

      logger.info("Please provide as argument:")
      logger.info("1) Path to label_index_info_test.")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Path to training log file.")
      logger.info("4) Target path for evaluation results.")
      sys.exit();

   labelIndexInfoPath = sys.argv[1]
   labelIndexMappingPath = sys.argv[2]
   logFilePath = sys.argv[3]
   evaluationTargetPath = sys.argv[4]
   if evaluationTargetPath.endswith('/') == False:
      evaluationTargetPath += '/'

   if not os.path.exists(os.path.dirname(evaluationTargetPath)) and os.path.dirname(evaluationTargetPath) != '':
      os.makedirs(os.path.dirname(evaluationTargetPath))


   plotTrainingLossAndAccuracy(logFilePath, evaluationTargetPath)

   # sys.exit()
   labels = utility.getLabelStrings(labelIndexMappingPath)
   [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = testNeuralNet(labels, labelIndexInfoPath, evaluationTargetPath)

   overviewPath = evaluationTargetPath + "overview.csv"
   logger.info('Writing file ' + overviewPath)
   np.savetxt(overviewPath, np.array([0,meanAveragePrecision, overallCorrect, overallWrong]), delimiter=',')

   confusionMatrixPath = evaluationTargetPath + 'confusionMatrix.npy'
   logger.info('Writing file ' + confusionMatrixPath)
   with open(confusionMatrixPath, "w") as outputFile:
      np.save(outputFile, confusionMatrix)

   utility.plotConfusionMatrix(confusionMatrix, labels, evaluationTargetPath + 'confusionMatrix.pdf')
