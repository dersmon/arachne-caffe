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

   data = data[data[:,0].argsort()]

   targetPath += "overview.csv"
   logger.info('Writing file ' + targetPath)
   np.savetxt(targetPath, data, delimiter=',')

def saveConfusionMatrix(confusionMatrix, targetPath):
   logger.info('Writing file ' + targetPath)
   with open(targetPath, "w") as outputFile:
      np.save(outputFile, confusionMatrix)

def runTest(clusters, testFeatures, labels, perLabel):

   labelCount = len(labels)
   neurons = activations.shape[1] - labelCount

   confusionMatrix = np.zeros((labelCount,labelCount))

   # split activations by label
   activationsByLabel = utility.splitTestFeaturesByLabel(testFeatures, labelCount)

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

         confusion[0,np.argmax(bestCluster[neurons:])] += 1

         if np.argmax(bestCluster[neurons:]) == np.argwhere(currentActivation[neurons:] == 1):
            overallCorrect += 1
         else:
            overallWrong += 1

         averagePrecision = 0
         relevant = 0
         sortedPredictions = None
         if perLabel == False:
            sortedPredictions = np.argsort(bestCluster[neurons:])[::-1].tolist()
         else:
            sortedPredictions = np.argmax(clusters[clusterRanking][:,neurons:], axis=1)
            # logger.debug(sortedPredictions)
         for idx, value in enumerate(sortedPredictions):
            indicator = 0
            if(value == np.argwhere(currentActivation[neurons:] == 1)):
               if perLabel == False and bestCluster[neurons + value] == 0: # ingore if 0 in histogram
                  continue
               indicator = 1
               relevant += 1

            precision = float(relevant) / (idx + 1)
            averagePrecision += (precision * indicator)


         if relevant != 0:
            averagePrecision = float(averagePrecision) / relevant
         meanAveragePrecision += averagePrecision

         activationCounter += 1

      confusionMatrix[labelIndex,:] = confusion

   meanAveragePrecision = float(meanAveragePrecision) / activations.shape[0]
   
   logger.info(' Accuracy: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))
   logger.info(' Mean average precision: '+str(meanAveragePrecision))

   return [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong]

def loadClusters(sourcePath):
   clusterSets = []
   for rootPath, subdirs, files in os.walk(sourcePath):
      for f in files:
         if f.endswith('clusters.npy'):

            split = rootPath.split('/')[::-1][1].split('_')
            splitLength = len(split)

            k = split[splitLength - 1]
            clusteringType = split[splitLength - 2]

            clusterSets.append({'file': f,'path':rootPath + '/', 'data':utility.arrayFromFile(rootPath + '/' +  f), 'k':int(k), 'type':clusteringType})

   return clusterSets

def runTests(activationsPath, indexLabelMappingPath, sourcePath):

   labels = utility.getLabelStrings(indexLabelMappingPath)
   activations = utility.arrayFromFile(activationsPath)

   if sourcePath.endswith('/') == False:
      sourcePath += '/'

   clusterGroups = loadClusters(sourcePath) # returns list with all previously generated cluster groups

   mixedClusterResultsSimple = []
   perLabelClusterResultsSimple = []

   for clusters in clusterGroups:
      logger.info("Evaluating clusters at " + clusters['path'])

      if(clusters['type'] == 'perLabel'):
         [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = runTest(clusters['data'], activations, labels True) # (clusters, activations, labels)
         perLabelClusterResultsSimple.append([clusters['k'], meanAveragePrecision, overallCorrect, overallWrong])
         saveConfusionMatrix(confusionMatrix, clusters['path'] + "confusion_test.npy")
         utility.plotConfusionMatrix(confusionMatrix, labels, clusters['path'] + "confusion_test.pdf")
      else:
         [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = runTest(clusters['data'], activations, labels False) # (clusters, activations, labels)
         mixedClusterResultsSimple.append([clusters['k'], meanAveragePrecision, overallCorrect, overallWrong])
         saveConfusionMatrix(confusionMatrix, clusters['path'] + "confusion_test.npy")
         utility.plotConfusionMatrix(confusionMatrix, labels, clusters['path'] + "confusion_test.pdf")

   overviewPerLabel = np.array(perLabelClusterResultsSimple)
   overviewMixed = np.array(mixedClusterResultsSimple)

   saveOverview(overviewPerLabel, sourcePath + 'perLabel_')
   saveOverview(overviewMixed, sourcePath + 'result_mixed_')

   utility.plotKMeansOverview(overviewPerLabel, sourcePath + 'perLabel_result.pdf', True)
   utility.plotKMeansOverview(overviewMixed, sourcePath + 'result_mixed_result.pdf', True)

if __name__ == '__main__':
   if len(sys.argv) != 4:

      logger.info("Please provide as argument:")
      logger.info("1) Path to test activations (*.npy).")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Source path for evaluation clusters.")
      sys.exit();

   runTests(sys.argv[1], sys.argv[2], sys.argv[3])
