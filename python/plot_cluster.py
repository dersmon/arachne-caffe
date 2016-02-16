import numpy as np
import random
import logging
import sys
import os
import modules.utility as utility

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plotPositions(positions, additionalRows, targetPath):
   positionsExtended = np.repeat(positions, [additionalRows]*positions.shape[0], axis=0)
   plt.imshow(positionsExtended, 'afmhot', interpolation='none')

   ax = plt.gca()
   ax.set_xlabel('Coordinates')
   ax.set_ylabel('Clusters')
   plt.savefig(targetPath + 'cluster_positions.pdf')
   plt.close()

def plotLabelGrid(clusterLabelHistograms, labelCount, labels, targetPath):
   clusterLabelHistograms = clusterLabelHistograms.T
   maxValue = np.max(clusterLabelHistograms, axis=1)

   ax = plt.gca()
   scaled = (clusterLabelHistograms.T / maxValue).T

   labelTicks = np.arange(0, labelCount * (scaled.shape[0] / labelCount))

   if labels != None and scaled.shape[0] > scaled.shape[1] * 0.75:
      labels = np.arange(0, labelCount)
      ax.set_yticks(labelTicks)
      ax.set_yticklabels(labels)

   ax.set_ylabel('Labels')
   ax.set_xlabel('Clusters')

   plt.imshow(scaled, 'Blues', interpolation='none')
   plt.savefig(targetPath + 'confusion_cluster.pdf', bbox_inches='tight')
   plt.close()

def plotClusters(clusters, iterations, neurons, targetPath):
   plotClusters(clusters, iterations, neurons, targetPath, None)

def plotClusters(clusters, iterations, neurons, targetPath, labels):
   logger.debug(clusters.shape)
   logger.debug(neurons)
   labelCount = clusters.shape[1] - neurons
   additionalRows = 1024 / clusters.shape[0]
   positions = clusters[:,0:neurons]
   clusterHistograms = clusters[:,neurons:]

   plotPositions(positions, additionalRows, targetPath)
   plotLabelGrid(clusterHistograms, labelCount, labels, targetPath)

if __name__ == '__main__':

   if len(sys.argv) != 5 and len(sys.argv) != 6:
      logger.info("Please provide as arguments:")
      logger.info("1) Cluster file (*.npy).")
      logger.info("2) Iterations file (*.npy)")
      logger.info("3) The the number of activation neurons.")
      logger.info("4) The target path.")
      logger.info("5) The path to the index label mapping (optional).")
      sys.exit()

   labels = None
   if len(sys.argv) == 6:
      labels = utility.getLabelStrings(sys.argv[5])

   clusters = utility.arrayFromFile(sys.argv[1])
   iterations = loadArray(sys.argv[2])
   neurons = int(sys.argv[3])

   targetPath = sys.argv[4]

   if targetPath.endswith('/') == False:
      targetPath += '/'

   evaluateClusters(clusters, iterations, neurons, targetPath, labels)
