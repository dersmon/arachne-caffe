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

def plotLabelGrid(clusterLabelHistograms, labelCount, labels, targetPath):
   clusterLabelHistograms = clusterLabelHistograms.T
   maxValue = np.max(clusterLabelHistograms, axis=1)

   ax = plt.gca()
   scaled = (clusterLabelHistograms.T / maxValue).T

   for(j,i),label in np.ndenumerate(clusterLabelHistograms):
      ax.text(i,j, int(label), ha='center', va='center', color='black',  fontsize=8)

   labelTicks = np.arange(labelCount)

   if labels == None:
      labels = np.arange(0,labelCount)

   ax.set_ylabel('Labels')
   ax.set_yticks(labelTicks)
   ax.set_yticklabels(labels)

   ax.set_xlabel('Clusters')
   ax.set_xticks(np.arange(0, clusterLabelHistograms.shape[1]))
   if len(labels) < clusterLabelHistograms.shape[1] / 2:
      ax.set_xticklabels(np.arange(0, clusterLabelHistograms.shape[1]), rotation=45, ha='right')
   else:
      ax.set_xticklabels(np.arange(0, clusterLabelHistograms.shape[1]))

   plt.imshow(scaled, 'Blues', interpolation='none')
   plt.savefig(targetPath + 'matrix.pdf', bbox_inches='tight')

def plotClusters(clusters, iterations, neurons, targetPath):
   evaluateClusters(clusters, iterations, neurons, targetPath, None)

def plotClusters(clusters, iterations, neurons, targetPath, labels):
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
