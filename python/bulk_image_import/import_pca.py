# ~encoding UTF-8

import numpy as np
import logging
import sys
import os

import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.arachne_caffe as ac

# 1 Mean Normalization
# 2 Sigma = 1 / m * X' * X, wobei m Anzahl der Datensaetze und X Matrix mit Datensaetzen
# 3 [U,S,V] = svg(Sigma)
# 3.1 1 - (sum(i->k):Sii/sum(i->n):Sii) fuer verschiedene K
# 4 Ureduce = U(:, 1:k)
# z = Ureduce' * x



logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SOURCE_DIMENSIONS = 4096

def meanNormalization(activations):
   logger.info("Normalizing input activations...")
   # logger.debug(activations.shape)

   activationMeans = np.mean(activations, 0)
   # logger.debug(activationMeans.shape)

   substractedMean = activations - activationMeans
   # logger.debug(substractedMean.shape)

   # logger.debug(activations[0][0])
   # logger.debug(activationMeans[0])
   # logger.debug(substractedMean[0][0])
   # logger.debug(activations[0][0] - activationMeans[0] == substractedMean[0][0])
   #
   # logger.debug(activations[50][1])
   # logger.debug(activationMeans[1])
   # logger.debug(substractedMean[50][1])
   # logger.debug(activations[4500][1] - activationMeans[1] == substractedMean[4500][1])

   standardDeviationMatrix = 1 / np.std(activations, 0)
   standardDeviationMatrix[standardDeviationMatrix >= 2147483648] = 0
   # logger.debug(standardDeviationMatrix.shape)
   # logger.debug(standardDeviationMatrix)



   activations = (substractedMean * standardDeviationMatrix)
   # logger.debug(activations.shape)

   # activations = (activations - np.amin(activations)) / (np.amax(activations) - np.amin(activations))

   return activations
def calculateCovariance(activations):

   logger.info("Calculating covariance...")
   # logger.debug(activations.shape)

   m = activations.shape[0]
   # logger.debug(m)

   sigma = np.dot(activations.T, activations) / m
   # logger.debug(sigma.shape)

   return sigma

def calculateEigenvectors(covarianceMatrix):
   logger.info("Calculating eigenvectors...")
   return np.linalg.svd(covarianceMatrix, full_matrices=True)

def findOptimalDimensionCount(singularValuesMatrix):
   logger.info("Searching viable dimensions to reduce...")

   k = 1
   while(k < singularValuesMatrix.shape[0]):
      kSum = np.sum(singularValuesMatrix[0:k])
      nSum = np.sum(singularValuesMatrix[:])

      result = 1 - (kSum/nSum)
      logger.debug(result)
      if(result > 0.01):
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

   # test = np.zeros(sourceActivations[:,0:SOURCE_DIMENSIONS].shape)
   # test[:,0:1000] = np.random.rand(test.shape[0],1000)
   #
   # logger.debug("Max: " + str(np.amax(test)) + ", min: " + str(np.amin(test)))
   # activations = meanNormalization(test)

   logger.debug("Max: " + str(np.amax(sourceActivations[:,0:SOURCE_DIMENSIONS])) + ", min: " + str(np.amin(sourceActivations[:,0:SOURCE_DIMENSIONS])))
   activations = meanNormalization(sourceActivations[:,0:SOURCE_DIMENSIONS])

   logger.debug("Max: " + str(np.amax(activations)) + ", min: " + str(np.amin(activations)))


   if(singularValuesMatrix == None):

      covarianceMatrix = calculateCovariance(activations)
      sys.exit
      unitaryMatrix, singularValuesMatrix, V = calculateEigenvectors(covarianceMatrix)

      singularValuesFile = os.path.splitext(sys.argv[1])[0] + "_singular.npy"
      singularValuesTextFile = os.path.splitext(sys.argv[1])[0] + "_singular.txt"

      if not os.path.exists(os.path.dirname(singularValuesFile)):
         os.makedirs(os.path.dirname(singularValuesFile))
      with open(singularValuesFile, "w") as outputFile:
         np.save(outputFile, singularValuesMatrix)

      unitaryMatrixFile = os.path.splitext(sys.argv[1])[0] + "_unitary_matrix.npy"

      if not os.path.exists(os.path.dirname(unitaryMatrixFile)):
         os.makedirs(os.path.dirname(unitaryMatrixFile))
      with open(unitaryMatrixFile, "w") as outputFile:
         np.save(outputFile, unitaryMatrix)

   # n = len(singularValuesMatrix)
   # # reverse the n first columns of u
   # unitaryMatrix[:,:n] = unitaryMatrix[:, n-1::-1]
   # # reverse s
   # singularValuesMatrix = singularValuesMatrix[::-1]
   # # reverse the n first rows of vt
   # # vt[:n, :] = vt[n-1::-1, :]

   # logger.debug(singularValuesMatrix)
   logger.debug(singularValuesMatrix.shape)
   logger.debug(unitaryMatrix.shape)

   k = findOptimalDimensionCount(singularValuesMatrix)
   logger.info("Optimal k: " + str(k))

   uReduced = unitaryMatrix[:,0:2048]
   logger.debug(uReduced.shape)
   logger.debug(activations.shape)
   z = np.dot(activations, uReduced)
   logger.debug(z.shape)

   # Reverse for testing

   activationsReconstructed = np.dot(uReduced, z.T).T
   logger.debug(activationsReconstructed.shape)

   logger.debug(activations[0,0])
   logger.debug(activationsReconstructed[0,0])

   logger.debug(activations[4500,0])
   logger.debug(activationsReconstructed[4500,0])

   logger.debug(activations[2100,2303])
   logger.debug(activationsReconstructed[2100,2303])
