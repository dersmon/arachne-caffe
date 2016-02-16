# ~encoding UTF-8

import numpy as np
import logging
import sys
import os

import json
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.arachne_caffe as ac

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# SOURCE_DIMENSIONS = 1183
SOURCE_DIMENSIONS = 4096
CREATE_RESTORED = False

PCA_SUBFOLDER_PATH = None
FILENAME_WITHOUT_EXTENSION = None

def meanNormalization(activations):
   logger.debug("Normalizing...")

   minValues = np.amin(activations, 0)
   maxValues = np.amax(activations, 0)

   result = (activations - minValues) / (maxValues - minValues)
   return result

def getSVU(activations, activationsSourcePath):

   logger.info('Calculating SVD...')

   U, s, V = np.linalg.svd(activations, full_matrices=False)
   logger.debug('full_matrices=False: ')
   logger.debug('U matrix shape: ' + str(U.shape))
   logger.debug('s vector shape: ' + str(s.shape))
   logger.debug('V matrix shape: ' + str(V.shape))

   singularValuesFile = PCA_SUBFOLDER_PATH + FILENAME_WITHOUT_EXTENSION + "_singular.npy"

   with open(singularValuesFile, "w") as outputFile:
      np.save(outputFile, s)

   unitaryMatrixFile = PCA_SUBFOLDER_PATH + FILENAME_WITHOUT_EXTENSION + "_unitary_matrix_V.npy"

   with open(unitaryMatrixFile, "w") as outputFile:
      np.save(outputFile, V)

   unitaryMatrixFile = PCA_SUBFOLDER_PATH + FILENAME_WITHOUT_EXTENSION + "_unitary_matrix_U.npy"

   with open(unitaryMatrixFile, "w") as outputFile:
      np.save(outputFile, U)

   return [s, V, U]

def findOptimalDimensionCount(singularValuesMatrix):
   logger.info("Searching viable k dimensions to reduce to...")

   k = 1
   nSum = np.sum(singularValuesMatrix)
   while(k < singularValuesMatrix.shape[0]):
      kSum = np.sum(singularValuesMatrix[0:k])
      result = 1 - (kSum/nSum)
      if(result <= 0.01):
         return k

      k += 1

   return k

if __name__ == '__main__':

   sourceActivations = None

   s = None
   U = None
   V = None
   trainingActivationsPath = ''
   if(len(sys.argv) != 3 and len(sys.argv) != 6):
      logger.info("Please provide as arguments:")
      logger.info("1) path to activations as npy file.")
      logger.info("2) K dimensions to omit (0 if algorithm should make decision).")
      logger.info("2) path to singular values matrix (*.npy) (optional).")
      logger.info("3) path to V Matrix (*.npy) (optional).")
      logger.info("4) path to U Matrix (*.npy) (optional)")
      logger.info("Please provide all matrixes or none.")
      sys.exit()

   logger.info("Expecting " + str(SOURCE_DIMENSIONS) + " source neurons.")

   trainingActivationsPath = sys.argv[1]

   PCA_SUBFOLDER_PATH = os.path.dirname(trainingActivationsPath) + "/pca/"
   FILENAME_WITHOUT_EXTENSION = os.path.splitext(trainingActivationsPath)[0].split('/')[-1]

   if not os.path.exists(os.path.dirname(PCA_SUBFOLDER_PATH)):
      os.makedirs(os.path.dirname(PCA_SUBFOLDER_PATH))

   logger.info("Path for matrices and restored activations: " + PCA_SUBFOLDER_PATH)

   k = int(sys.argv[2])
   sourceActivations = ac.activationsFromFile(trainingActivationsPath)
   if(len(sys.argv) == 6):
      with open(sys.argv[3], 'r') as inputFile:
         s = np.load(inputFile)
      with open(sys.argv[4], 'r') as inputFile:
         V = np.load(inputFile)
      with open(sys.argv[5], 'r') as inputFile:
         U = np.load(inputFile)

   if(sourceActivations == None):
      logger.error("Could not load activations, exiting.")
      sys.exit()

   activations = sourceActivations[:,0:SOURCE_DIMENSIONS]
   logger.info('Original shape: ' + str(activations.shape))

   if(s == None):
      [s,V,U] = getSVU(activations, trainingActivationsPath)

   if k == 0:
      k = findOptimalDimensionCount(s)
      logger.info("Found k: " + str(k))

   S = np.diag(s[:k])


   reduced = np.dot(S, U[:,:k].T).T
   logger.info('Reduced shape: ' + str(reduced.shape))
   restored = None
   if CREATE_RESTORED:
      restored = np.dot(reduced, V[:k,:])
   # logger.debug('restored: ' + str(restored.shape))

   absDiffMatrix = np.absolute(activations - restored)

   # logger.debug('Sum original activations: ' + str(np.sum(np.absolute(activations))))
   # logger.debug('Sum restored activations: ' + str(np.sum(np.absolute(restored))))

   absDiffMatrix = np.absolute(activations - restored)
   # logger.debug('Median in np.absolute(activations - restored): ' + str(np.median(absDiffMatrix)))
   # logger.debug('Min/max value in np.absolute(activations - restored): ' + str(np.amin(absDiffMatrix))+ '/' + str(np.amax(absDiffMatrix)))
   # logger.debug('Sum np.absolute(activations - restored): ' + str(np.sum(absDiffMatrix)))

   reducedActivationsPath = os.path.splitext(trainingActivationsPath)[0] + "_reduced_" + str(reduced.shape[1]) + ".npy"
   reduced = np.hstack((reduced, sourceActivations[:,SOURCE_DIMENSIONS:]))

   logger.info('Writing file: ' + reducedActivationsPath)
   if not os.path.exists(os.path.dirname(reducedActivationsPath)):
      os.makedirs(os.path.dirname(reducedActivationsPath))
   with open(reducedActivationsPath, "w") as outputFile:
      np.save(outputFile, reduced)

   if restored != None:
      restoredActivationsPath = PCA_SUBFOLDER_PATH + FILENAME_WITHOUT_EXTENSION + "_restored_" + str(restored.shape[1]) + ".npy"
      restored = np.hstack((restored, sourceActivations[:,SOURCE_DIMENSIONS:]))

      logger.info('Writing file: ' + restoredActivationsPath)
      with open(restoredActivationsPath, "w") as outputFile:
         np.save(outputFile, restored)
