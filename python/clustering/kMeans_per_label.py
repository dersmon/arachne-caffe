import numpy as np
import random
import logging
import sys
import os
import kMeans_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.utility as utility

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS = 150

def findKMeansPerLabel(activations, k, labelCount, targetPath):
   findKMeansPerLabel(activations, k, labelCount, targetPath, None)

def findKMeansPerLabel(activations, k, labelCount, targetPath, indexLabelMapping):
   labels = None
   if indexLabelMapping != None:
      labels = utility.getLabelStrings(indexLabelMapping)

   logger.debug(labelCount)

   # split activations by label
   activationsByLabel = []
   counter = 0
   while counter < labelCount:
      currentLabelIndex = activations.shape[1] - labelCount + counter
      logger.debug(currentLabelIndex)
      currentSelection = activations[activations[:, currentLabelIndex] == 1]
      activationsByLabel.append(currentSelection)
      counter += 1

   logger.debug("Activations shape: " + str(activations.shape))
   logger.debug("Activations by label length: " + str(len(activationsByLabel)))
   logger.debug("Activations by label 0 shape: " + str(activationsByLabel[0].shape))

   counter = 0
   clusters = []
   iterations = []
   for batch in activationsByLabel:
      if labels != None:
         logger.info('Running KMeans for label ' + labels[counter] + '.')
      else:
         logger.info('Running KMeans for label ' + str(counter))
      logger.debug("Batch shape: " + str(batch.shape))
      [c, i] = kMeans_core.runKMeans(batch, labelCount, k, MAX_ITERATIONS)
      clusters.extend(c)
      iterations.append(i)
      counter += 1

   kMeans_core.saveResults(clusters, iterations, targetPath)

   return [clusters, iterations]

if __name__ == '__main__':

   if len(sys.argv) != 5 and len(sys.argv) != 6:
      logger.info("Please provide as argument:")
      logger.info("1) npy-file with activations.")
      logger.info("2) K per label.")
      logger.info("3) The the number of activation neurons.")
      logger.info("4) The target path.")
      logger.info("5) The path to the index label mapping (optional).")
      sys.exit()

   activations = None
   logger.info('Opening file: ' + sys.argv[1])
   with open(sys.argv[1], 'r') as inputFile:
      activations = np.load(inputFile)

   k = int(sys.argv[2])
   labelCount = len(activations[0][int(sys.argv[3]):])

   targetPath = sys.argv[4]
   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   if len(sys.argv) == 6:
      findKMeansPerLabel(activations, k, labelCount, targetPath, sys.argv[5])
   else:
      findKMeansPerLabel(activations, k, labelCount, targetPath)
