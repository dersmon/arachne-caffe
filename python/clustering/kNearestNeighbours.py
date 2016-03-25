import numpy as np
from scipy import sparse
import resource
import sys
import types

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
BATCH_SIZE = 1000

def getKNearestNeighbours(training, test, k, neurons):

   results = np.empty((test.shape[0], test.shape[1] - neurons + k))

   logger.info("Searching " + str(k) + " nearest neighbours.")

   counter = 0
   while counter < test.shape[0]:
      currentTestVector = test[counter]

      differences = training[:,:neurons] - currentTestVector[:neurons]
      squaredDistances = np.linalg.norm(differences, None, 1)
      notZero = squaredDistances != 0
      finalDistances = 1 / squaredDistances[notZero]

      neighbourIds = np.argsort(finalDistances)[::-1][:k]

      results[counter,:] = np.hstack((currentTestVector[neurons:], neighbourIds))
      counter += 1
   logger.info('Calculated ' + str(k) +' nearest neighbours for ' + str(counter) + ' test vectors.')
   
   return results
