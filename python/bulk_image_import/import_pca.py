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
logger.setLevel(logging.DEBUG)

SOURCE_DIMENSIONS = 4096

def meanNormalization(activations):
   logger.info("Normalizing input activations...")

   # activationMeans = np.mean(activations, 0)
   #
   # substractedMean = activations - activationMeans
   #
   # standardDeviationMatrix = np.std(activations, 0)
   #
   # logger.debug(activations.shape)
   # logger.debug(standardDeviationMatrix.shape)
   # logger.debug(substractedMean.shape)
   #
   # result = (substractedMean / standardDeviationMatrix)

   minValues = np.amin(activations, 0)
   maxValues = np.amax(activations, 0)

   # logger.debug(minValues.shape)
   # logger.debug(maxValues.shape)
   # logger.debug(activations.shape)

   result = (activations - minValues) / (maxValues - minValues)
   return result
def calculateCovariance(activations):

   logger.info("Calculating covariance...")
   m = activations.shape[0]
   sigma = np.dot(activations.T, activations) / m

   return sigma

def calculateEigenvectors(covarianceMatrix):
   logger.info("Calculating eigenvectors...")
   return np.linalg.svd(covarianceMatrix, full_matrices=True)

def findOptimalDimensionCount(singularValuesMatrix):
   logger.info("Searching viable dimensions to reduce...")

   k = 1
   nSum = np.sum(singularValuesMatrix[:])

   matrixReversed = singularValuesMatrix[::-1]

   while(k < singularValuesMatrix.shape[0]):
      kSum = np.sum(matrixReversed[0:k])

      result = 1 - (kSum/nSum)
      if(result < 0.99):
         if(k == 1):
            return k
         return k - 1

      k += 1

   return k

if __name__ == '__main__':

   sourceActivations = None
   singularValuesMatrix = None
   unitaryMatrix = None

   if(len(sys.argv) == 1 or len(sys.argv) > 4):
      logger.info("Please provide as arguments:")
      logger.info("1) path to activations as npy file.")
      logger.info("2) path to singular values matrix as npy file (optional)")
      sys.exit
   else:
      sourceActivations = ac.activationsFromFile(sys.argv[1])
      if(len(sys.argv) == 4):
         with open(sys.argv[2], 'r') as inputFile:
            singularValuesMatrix = np.load(inputFile)
         with open(sys.argv[3], 'r') as inputFile:
            unitaryMatrix = np.load(inputFile)

   if(sourceActivations == None):
      logger.error("Could not load activations, exiting.")
      sys.exit

   logger.debug("Max: " + str(np.amax(sourceActivations[:,0:SOURCE_DIMENSIONS])) + ", min: " + str(np.amin(sourceActivations[:,0:SOURCE_DIMENSIONS])))
   activations = meanNormalization(sourceActivations[:,0:SOURCE_DIMENSIONS])

   logger.debug("Max: " + str(np.amax(activations)) + ", min: " + str(np.amin(activations)) + " (Normalized)")


   if(singularValuesMatrix == None):

      covarianceMatrix = calculateCovariance(activations)
      sys.exit
      unitaryMatrix, singularValuesMatrix, V = calculateEigenvectors(covarianceMatrix)

      singularValuesFile = os.path.splitext(sys.argv[1])[0] + "_singular.npy"

      if not os.path.exists(os.path.dirname(singularValuesFile)):
         os.makedirs(os.path.dirname(singularValuesFile))
      with open(singularValuesFile, "w") as outputFile:
         np.save(outputFile, singularValuesMatrix)

      unitaryMatrixFile = os.path.splitext(sys.argv[1])[0] + "_unitary_matrix.npy"

      if not os.path.exists(os.path.dirname(unitaryMatrixFile)):
         os.makedirs(os.path.dirname(unitaryMatrixFile))
      with open(unitaryMatrixFile, "w") as outputFile:
         np.save(outputFile, unitaryMatrix)

   k = findOptimalDimensionCount(singularValuesMatrix)
   logger.info("Optimal k: " + str(k))

   uReduced = unitaryMatrix[:,0:activations.shape[1]-k]
   # uReduced = unitaryMatrix[:,0:4000]
   z = np.dot(activations, uReduced)
   #
   # activationsReconstructed = np.dot(uReduced, z.T).T
   #
   # logger.debug(np.allclose(activations, activationsReconstructed))
   # logger.debug(np.sum(activations))
   # logger.debug(np.sum(activationsReconstructed ))
   #
   # absDiffMatrix = np.absolute(activations - activationsReconstructed)
   # logger.debug('-----')
   # logger.debug(np.median(absDiffMatrix))
   # logger.debug(np.amax(absDiffMatrix))
   # logger.debug(np.amin(absDiffMatrix))
   # logger.debug(np.sum(absDiffMatrix))
   #
   # plt.hist(absDiffMatrix.flatten(), 50)
   # plt.show()
