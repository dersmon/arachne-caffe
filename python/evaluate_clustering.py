import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging

import modules.utility as utility
import clustering.kMeans_core as kMeans_core

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def saveOverview(data, targetPath):
   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   data = data[data[:,0].argsort()]

   targetPath += "overview.csv"
   logger.info('Writing file ' + targetPath)
   np.savetxt(targetPath, data, delimiter=',')

def saveConfusionMatrix(confusionMatrix, targetPath):
   logger.info('Writing file ' + targetPath)
   with open(targetPath, "w") as outputFile:
      np.save(outputFile, confusionMatrix)

def getSumAndFactorPerLabel(clusters, neurons):
   allLabels = clusters[:,neurons:]
   sumPerLabel = np.sum(allLabels, axis=0) / allLabels.shape[0]
   factorPerLabel = np.max(sumPerLabel) / sumPerLabel
   return [sumPerLabel, factorPerLabel]

def runTest(clusters, activations, labels, factorPerLabel):

   labelCount = len(labels)
   neurons = activations.shape[1] - labelCount

   confusionMatrix = np.zeros((labelCount,labelCount))

   # split activations by label
   activationsByLabel = [[] for i in range(labelCount)]
   counter = 0
   while counter < labelCount:
      currentLabelIndex = activations.shape[1] - labelCount + counter
      currentSelection = activations[activations[:, currentLabelIndex] == 1]
      activationsByLabel[counter] = currentSelection
      counter += 1

   overallCorrect = 0
   overallWrong = 0
   meanAveragePrecision = 0
   # evaluate
   for labelIndex, values in enumerate(activationsByLabel):
      confusion = np.zeros((1,len(labels)))

      activationCounter = 0
      while activationCounter < values.shape[0]:
         currentActivation = values[activationCounter]
         clusterRanking = kMeans_core.predictSimple(clusters, currentActivation, neurons)
         bestCluster = clusters[clusterRanking[0]]
         if factorPerLabel != None:
            labelHistogram = bestCluster[neurons:]
            labelHistogram = np.multiply(bestCluster[neurons:], factorPerLabel)
            bestCluster[neurons:] = labelHistogram

         confusion[0,np.argmax(bestCluster[neurons:])] += 1

         if np.argmax(bestCluster[neurons:]) == np.argwhere(currentActivation[neurons:] == 1):
            overallCorrect += 1
            # correctPerLabel[np.argwhere(activation[neurons:] == 1)] += 1
         else:
            overallWrong += 1
            # wrongPerLabel[np.argwhere(activation[neurons:] == 1)] += 1

         averagePrecision = 0
         relevant = 0

         sortedHistogramIndices = np.argsort(bestCluster[neurons:])[::-1].tolist()
         for idx, value in enumerate(sortedHistogramIndices):
            indicator = 0
            if(value == np.argwhere(currentActivation[neurons:] == 1) and value != 0):
               indicator = 1
               relevant += 1

            precision = float(relevant) / (idx + 1)
            averagePrecision += (precision * indicator)

         if relevant != 0:
            averagePrecision = float(averagePrecision) / relevant
         meanAveragePrecision += averagePrecision

         activationCounter += 1

      confusionMatrix[labelIndex,:] = confusion


   logger.info(' Accuracy: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))
   meanAveragePrecision = float(meanAveragePrecision) / activations.shape[0]
   logger.info(' Mean average precision: '+str(meanAveragePrecision))

   return [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong]

def runTests(activationsPath, indexLabelMappingPath, sourcePath, sourcePrefix):

   labels = utility.getLabelStrings(indexLabelMappingPath)
   activations = utility.arrayFromFile(activationsPath)

   if sourcePath.endswith('/') == False:
      sourcePath += '/'

   clusterSets = []
   for rootPath, subdirs, files in os.walk(sourcePath):
      # logger.debug(rootPath)

      # logger.debug('Correct root: ' + rootPath)
      for f in files:
         if f.endswith('clusters.npy'):
            # logger.debug('Found clusters file: ' + f)

            # Type and k are parsed from root folder ending.
            split = rootPath.split('/')[::-1][1].split('_')
            splitLength = len(split)
            logger.debug(split)
            k = split[splitLength - 1]
            clusteringType = split[splitLength - 2]

            logger.debug(k)
            logger.debug(clusteringType)

            clusterSets.append({'file': f,'path':rootPath + '/', 'data':utility.arrayFromFile(rootPath + '/' +  f), 'k':int(k), 'type':clusteringType})

   mixedClusterResultsSimple = []
   mixedClusterResultsSimpleNormalized = []
   perLabelClusterResultsSimple = []

   for clusters in clusterSets:
      [sumPerLabel, factorPerLabel] = getSumAndFactorPerLabel(clusters['data'], activations.shape[1] - len(labels))

      logger.info("Evaluating clusters at " + clusters['path'])

      if(clusters['type'] == 'perLabel'):
         [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = runTest(clusters['data'], activations, labels, None) # (clusters, activations, labels)
         perLabelClusterResultsSimple.append([clusters['k'], meanAveragePrecision, overallCorrect, overallWrong])
         saveConfusionMatrix(confusionMatrix, clusters['path'] + "confusion_test.npy")
         utility.plotConfusionMatrix(confusionMatrix, labels, clusters['path'] + "confusion_test.pdf")
      else:
         [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = runTest(clusters['data'], activations, labels, None) # (clusters, activations, labels)
         mixedClusterResultsSimple.append([clusters['k'], meanAveragePrecision, overallCorrect, overallWrong])
         saveConfusionMatrix(confusionMatrix, clusters['path'] + "confusion_test.npy")
         utility.plotConfusionMatrix(confusionMatrix, labels, clusters['path'] + "confusion_test.pdf")

         [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = runTest(clusters['data'], activations, labels, factorPerLabel) # (clusters, activations, labels)
         mixedClusterResultsSimpleNormalized.append([clusters['k'], meanAveragePrecision, overallCorrect, overallWrong])
         saveConfusionMatrix(confusionMatrix, clusters['path'] + "confusion_test_normalized.npy")
         utility.plotConfusionMatrix(confusionMatrix, labels, clusters['path'] + "confusion_test_normalized.pdf")

   overviewPerLabel = np.array(perLabelClusterResultsSimple)
   overviewMixed = np.array(mixedClusterResultsSimple)
   overviewMixedNormalized = np.array(mixedClusterResultsSimpleNormalized)

   saveOverview(overviewPerLabel, sourcePath + 'perLabel/')
   saveOverview(overviewMixed, sourcePath + 'result_mixed/')
   saveOverview(overviewMixedNormalized, sourcePath + 'result_mixed_normalized/')

   utility.plotKMeansOverview(overviewPerLabel, sourcePath + 'perLabel_result.pdf', True)
   utility.plotKMeansOverview(overviewMixed, sourcePath + 'result_mixed_result.pdf', True)
   utility.plotKMeansOverview(overviewMixedNormalized, sourcePath + 'result_mixed_normalized_result.pdf', True)

if __name__ == '__main__':
   if len(sys.argv) != 5:

      logger.info("Please provide as argument:")
      logger.info("1) Path to test activations (*.npy).")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Source path for evaluation clusters.")
      logger.info("4) Prefix for evaluation folders.")
      sys.exit();

   runTests(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
