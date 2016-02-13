import numpy as np
import random
import logging
import sys
import os
import kMeans_core

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS = 150

def findKMeans(activations, k, labelCount, targetPath):




   clusters, iterations = kMeans_core.runKMeans(activations, labelCount, k, MAX_ITERATIONS)

   kMeans_core.saveResults(clusters, iterations, targetPath + 'mixed_' + str(k))

if __name__ == '__main__':

   if len(sys.argv) != 4 and len(sys.argv) != 5:
      logger.info("Please provide as argument:")
      logger.info("1) npy-file with activations.")
      logger.info("2) K.")
      logger.info("3) The the number of activation neurons.")
      logger.info("4) The target path.")
      logger.info("5) The path to the index label mapping (optional).")
      sys.exit()

   activations = None
   logger.info('Opening file: ' + sys.argv[1])
   with open(sys.argv[1], 'r') as inputFile:
      activations = np.load(inputFile)

   targetPath = sys.argv[4]
   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   k = int(sys.argv[2])
   labelCount = len(activations[0][int(sys.argv[3]):])

   findKMeans(activations, k, labelCount, targetPath)
