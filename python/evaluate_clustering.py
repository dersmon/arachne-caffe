import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import modules.utility as utility
import clustering.kMeans_core as kMeans_core

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def testSimple(clusters, activations, labels):

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
            if(value == np.argwhere(currentActivation[neurons:] == 1)):
               indicator = 1
               relevant += 1

            precision = float(relevant) / (idx + 1)
            averagePrecision += (precision * indicator)

         averagePrecision = float(averagePrecision) / relevant
         meanAveragePrecision += averagePrecision

         activationCounter += 1

      confusionMatrix[labelIndex,:] = confusion


   logger.info('Exact prediction:')
   logger.info('correct: ' + str(overallCorrect) + ', wrong: ' + str(overallWrong) + ', ratio: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))

   meanAveragePrecision = float(meanAveragePrecision) / activations.shape[0]
   logger.info('Mean average precision:')
   logger.info(meanAveragePrecision)

   return [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong]


def runTests(activationsPath, indexLabelMappingPath, sourcePath, sourcePrefix):

   labels = utility.getLabelStrings(indexLabelMappingPath)
   activations = utility.arrayFromFile(activationsPath)

   if sourcePath.endswith('/') == False:
      sourcePath += '/'

   clusters = []
   for root, subdirs, files in os.walk(sourcePath):
      if root.split('/')[len(root.split('/'))-1].startswith(sourcePrefix):
         logger.debug('Correct root: ' + root)
         for f in files:
            if f.endswith('_clusters.npy'):
               logger.debug('Found clusters file: ' + f)
               clusters.append({'path':root + '/' +  f, 'data':utility.arrayFromFile(root + '/' +  f)})

   for cluster in clusters:

      [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = testSimple(cluster['data'], activations, labels) # (clusters, activations, labels)
      utility.plotConfusionMatrix(confusionMatrix, labels, cluster['path'])

if __name__ == '__main__':
   if len(sys.argv) != 5:

      logger.info("Please provide as argument:")
      logger.info("1) Path to test activations (*.npy).")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Source path for evaluation clusters.")
      logger.info("4) Prefix for evaluation folders.")
      sys.exit();

   runTests(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
