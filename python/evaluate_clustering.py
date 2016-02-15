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

def plotOverview(data, targetPath):

   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))
   # [cluster['k'], meanAveragePrecision, overallCorrect, overallWrong]
   counter = 0
   data = data[data[:,0].argsort()]
   logger.debug(data.shape)
   logger.debug(data)
   k = data[:,0]

   meanaverageprecision = data[:,1]
   correct = data[:,2] / (data[:,2] + data[:,3])

   fig, ax = plt.subplots()

   ax.plot(k, correct, 'gx', k, meanaverageprecision, 'bo')

   ax.set_xlabel('K')
   ax.axis([k[0], k[::-1][0], 0, 1])
   ax.grid(True)

   labelTrainingLoss = mpatches.Patch(color='g', label='Accuracy')
   labelTestLoss = mpatches.Patch(color='b', label='Mean average precision')

   plt.legend(handles=[labelTrainingLoss, labelTestLoss])
   plt.savefig(targetPath + 'result.pdf', bbox_inches='tight')
   plt.close()


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
   for rootPath, subdirs, files in os.walk(sourcePath):
      if rootPath.split('/')[len(rootPath.split('/'))-1].startswith(sourcePrefix):
         logger.debug('Correct root: ' + rootPath)
         for f in files:
            if f.endswith('clusters.npy'):
               logger.debug('Found clusters file: ' + f)

               # Type and k are parsed from root folder ending.
               split = rootPath.split('/')[::-1][0].split('_')
               splitLength = len(split)
               logger.debug(split)
               k = split[splitLength - 1]
               clusteringType = split[splitLength - 2]

               logger.debug(k)
               logger.debug(clusteringType)

               clusters.append({'file': f,'path':rootPath + '/', 'data':utility.arrayFromFile(rootPath + '/' +  f), 'k':int(k), 'type':clusteringType})

   mixedClusterResultsSimple = []
   perLabelClusterResultsSimple = []

   for cluster in clusters:

      logger.info("Evaluating clusters at " + cluster['path'])
      [confusionMatrix, meanAveragePrecision, overallCorrect, overallWrong] = testSimple(cluster['data'], activations, labels) # (clusters, activations, labels)

      if(cluster['type'] == 'perLabel'):
         perLabelClusterResultsSimple.append([cluster['k'], meanAveragePrecision, overallCorrect, overallWrong])
      else:
         mixedClusterResultsSimple.append([cluster['k'], meanAveragePrecision, overallCorrect, overallWrong])

      utility.plotConfusionMatrix(confusionMatrix, labels, cluster['path'] + "confusion_test.pdf")

   plotOverview(np.array(mixedClusterResultsSimple), sourcePath + 'result_mixed/')
   plotOverview(np.array(perLabelClusterResultsSimple), sourcePath + 'perLabel/')
   # plot overview results

if __name__ == '__main__':
   if len(sys.argv) != 5:

      logger.info("Please provide as argument:")
      logger.info("1) Path to test activations (*.npy).")
      logger.info("2) Path to label_index_mapping.")
      logger.info("3) Source path for evaluation clusters.")
      logger.info("4) Prefix for evaluation folders.")
      sys.exit();

   runTests(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
