import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import modules.utility as utility

import clustering.kMeans_core as kMeans_core
import clustering.kMeans_mixed as kMeans_mixed
import clustering.kMeans_per_label as kMeans_per_label

import plot_cluster as plot_cluster

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS_PER_LABEL = 150
MAX_ITERATIONS_MIXED = 150

PER_LABEL_START = 1
PER_LABEL_END = 12

MIXED_START = 5
MIXED_STEP = 5
MIXED_END = 120


def testClusters(clusters, activations, labels, targetPath):

   labelCount = len(labels)
   neurons = activations.shape[1] - labelCount

   imagesByLabel = [[] for i in range(len(labels))]
   confusionMatrix = np.zeros((len(labels),len(labels)))

   # split info by label
   activationsByLabel = []
   counter = 0
   while counter < labelCount:
      currentLabelIndex = activations.shape[1] - labelCount + counter
      logger.debug(currentLabelIndex)
      currentSelection = activations[activations[:, currentLabelIndex] == 1]
      activationsByLabel.append(currentSelection)
      counter += 1


   # create batches per split if nessecary
   for labelIndex, values in enumerate(activationsByLabel):
      confusion = np.zeros((1,len(labels)))

      distances = []

      for cluster in clusters:
         difference = np.tile(cluster['position'], (values.shape[0], 1)) - values[:,0:neurons]
         distances.append(np.linalg.norm(difference))

         # logger.debug(np.argsort(distances))

         sortedIndices = np.argsort(distances)
         confusion[0,np.argmax(clusters[sortedIndices[0]]['memberLabelHistogram'])] += 1

      # append sums to matrix
      confusionMatrix[labelIndex,:] = confusion

   utility.plotConfusionMatrix(confusionMatrix, labels, targetPath)

   # correct = 0
   # wrong = 0
   #
   # correctRanked = 0
   # wrongRanked = 0

   #
   #
   # tempCenters = []
   # updatedCenters = []
   #
   #
   #
   # correctPerLabel = [0] * labelCount
   # wrongPerLabel = [0] * labelCount
   #
   # meanAveragePrecision = 0
   # for activation in testVectors:
   #    distances = []
   #
   #    # Calculate distance to centers
   #    for center in clusters:
   #       difference = center['position'] - activation[0:neurons]
   #       distances.append(np.linalg.norm(difference))
   #
   #       # logger.debug(distances)
   #       # logger.debug(np.argsort(distances))
   #
   #    sortedIndices = np.argsort(distances)
   #
   #    centerCounter = 0
   #    if clusters[sortedIndices[0]]['maxLabelID'] == np.argwhere(activation[neurons:] == 1):
   #       correct += 1
   #       correctPerLabel[np.argwhere(activation[neurons:] == 1)] += 1
   #    else:
   #       wrong += 1
   #       wrongPerLabel[np.argwhere(activation[neurons:] == 1)] += 1
   #
   #    maxLabel = 0
   #    sumCorrectLabel = 0
   #
   #    # logger.debug(clusters[sortedIndices[0]]['labelDistribution'])
   #
   #
   #    for idx, labelCount in enumerate(clusters[sortedIndices[0]]['labelDistribution']):
   #       if labelCount > maxLabel:
   #          maxLabel = labelCount
   #       if(idx == np.argwhere(activation[neurons:] == 1)):
   #          sumCorrectLabel += labelCount
   #
   #    averagePrecision = 0
   #    relevant = 0
   #
   #    # logger.debug(clusters[sortedIndices[0]]['labelDistribution'])
   #    sortedHistogramIndices = np.argsort(clusters[sortedIndices[0]]['labelDistribution'])[::-1].tolist()
   #    # logger.debug(sortedHistogramIndices)
   #    for idx, value in enumerate(sortedHistogramIndices):
   #       # logger.debug(idx)
   #       # logger.debug(labelCount)
   #       indicator = 0
   #       if(value == np.argwhere(activation[neurons:] == 1)):
   #          indicator = 1
   #          relevant += 1
   #
   #       precision = float(relevant) / (idx + 1)
   #       averagePrecision += (precision * indicator)
   #
   #       # logger.debug('relevant: ' + str(indicator))
   #       # logger.debug('indicator: ' + str(indicator))
   #       # logger.debug('precision: ' + str(averagePrecision))
   #
   #    averagePrecision = float(averagePrecision) / relevant
   #    # logger.debug('average precision: ' + str(averagePrecision))
   #
   #    meanAveragePrecision += averagePrecision
   #    correctRanked += (float(sumCorrectLabel) / maxLabel)
   #
   #
   # logger.info('Exact prediction:')
   # logger.info('correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(float(correct)/(wrong + correct)))
   # logger.info('correct per label: ' + str(correctPerLabel))
   # logger.info('wrong per label: ' + str(wrongPerLabel))
   #
   # logger.info('Correct ranked: ')
   # logger.info(correctRanked / len(testVectors))
   #
   # meanAveragePrecision = float(meanAveragePrecision) / testVectors.shape[0]
   # logger.info('Mean average precision:')
   # logger.info(meanAveragePrecision)

def evaluateKMeans(trainingActivationsPath, testActivationsPath, labelIndexMappingPath, targetFolder, subfolderPrefix):
   if targetFolder.endswith('/') == False:
      targetFolder += '/'

   trainingActivations = utility.arrayFromFile(trainingActivationsPath)
   testActivations = utility.arrayFromFile(testActivationsPath)
   labels = utility.getLabelStrings(labelIndexMappingPath)
   neurons = trainingActivations.shape[1] - len(labels)
   logger.debug(labels)

   # test per label k-means
   k = PER_LABEL_START
   while k < PER_LABEL_END:
      logger.info("Testing per label KMeans with k = " + str(k) + ".")
      currentTarget = targetFolder + subfolderPrefix + "_perLabel_" + str(k) + "/"
      if not os.path.exists(os.path.dirname(currentTarget)):
         os.makedirs(os.path.dirname(currentTarget))

      c, i = kMeans_per_label.findKMeansPerLabel(trainingActivations, k, len(labels), currentTarget, labelIndexMappingPath)
      plot_cluster.plotClusters(kMeans_core.cleanUp(c), i, neurons, currentTarget, labels[:len(labels)])
      testClusters(c, testActivations, labels, currentTarget)

      k += 1

   # test mixed k-means
   k = MIXED_START
   while k < MIXED_END:
      logger.info("Testing mixed KMeans with k = " + str(k) + ".")
      currentTarget = targetFolder + subfolderPrefix + "_mixed_" + str(k) + "/"
      if not os.path.exists(os.path.dirname(currentTarget)):
         os.makedirs(os.path.dirname(currentTarget))
      [c,i] = kMeans_mixed.findKMeans(trainingActivations, k, len(labels), currentTarget)
      plot_cluster.plotClusters(kMeans_core.cleanUp(c), i, trainingActivations.shape[1] - len(labels), currentTarget, labels[:len(labels)])

      k += MIXED_STEP

if __name__ == '__main__':
   if len(sys.argv) != 6:

      logger.info("Please provide as argument:")
      logger.info("1) Path to training activations.")
      logger.info("2) Path to test activations.")
      logger.info("3) Path to label_index_mapping.")
      logger.info("4) Target path for evaluation results.")
      logger.info("5) Prefix for result folders.")
      sys.exit();

   evaluateKMeans(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
