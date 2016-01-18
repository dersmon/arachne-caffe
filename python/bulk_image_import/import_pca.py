# ~encoding UTF-8

import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.arachne_caffe as ac

# 1 Mean Normalization
# 2 Sigma = 1 / m * X' * X, wobei m Anzahl der Datensaetze und X Matrix mit Datensaetzen
# 3 [U,S,V] = svg(Sigma)
# 4 Ureduce = U(:, 1:k)
# z = Ureduce' * x

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SOURCE_DIMENSIONS = 4096

def meanNormalization(activations):
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
   # logger.debug(activations[0][1])
   # logger.debug(activationMeans[1])
   # logger.debug(substractedMean[0][1])
   # logger.debug(activations[0][1] - activationMeans[1] == substractedMean[0][1])

   standardDeviationMatrix = np.std(substractedMean, 0)
   # logger.debug(standardDeviationMatrix.shape)

   activations = (substractedMean / standardDeviationMatrix)
   # logger.debug(activations.shape)

   return activations

def calculateSigma(activations):
   logger.debug(activations.shape)
      
   m = activations.shape[0]
   logger.debug(m)

   sigma = np.dot(activations.T, activations) / m
   logger.debug(sigma.shape)

   return sigma

if __name__ == '__main__':
   sourcePath = ""
   originalActivations = None
   if(len(sys.argv) != 2):
      logger.info("Please provide as argument:")
      logger.info("1) path to activations as npy file.")
      sys.exit
   else:
      sourceActivations = ac.activationsFromFile(sys.argv[1])

   if(sourceActivations == None):
      logger.error("Could not load activations, exiting.")
      sys.exit

   activations = meanNormalization(sourceActivations[:,0:SOURCE_DIMENSIONS])
   sigma = calculateSigma(activations)
