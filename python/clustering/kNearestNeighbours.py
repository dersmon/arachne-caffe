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

   # input: training (7xxx,4096), test (18xx,4096)
   # output: (18xx, k)

   results = np.empty((test.shape[0], test.shape[1] - neurons + k))

   logger.info("Searching " + str(k) + " nearest neighbours.")# Batch size is " + str(batchSize) + ".")
   # previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

   counter = 0
   while counter < test.shape[0]:
      currentTestVector = test[counter]

      # logger.debug("Current test vector shape: " + str(currentTestVector.shape))
      #neighbourIds = np.empty(k)
      # batchCounter = 0

      # print ('1) Memory usage: %s (kb)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - previous))
      # previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

      # while batchCounter < k:
      #    difference  = training[batchCounter:batchCounter+BATCH_SIZE] - currentTestVector[0:4096]
      #    batchResult = np.vstack((np.linalg.norm(currentBatch[:,0:4096], axis=1), currentBatch[:,4096:].T))
      #
      #    distances[batchCounter:batchCounter+batchSize,:] = batchResult.T
      #
      #    batchCounter += BATCH_SIZE

      differences = training[:,:neurons] - currentTestVector[:neurons]
      # logger.debug("Differences shape: " + str(differences.shape))
      squaredDistances = np.linalg.norm(differences, None, 1)
      # logger.debug("Distances shape: " + str(squaredDistance.shape))
      notZero = squaredDistances != 0
      finalDistances = 1 / squaredDistances[notZero]
      # finalDistances = squaredDistances
      # logger.debug("Distances shape: " + str(finalDistances.shape))

      neighbourIds = np.argsort(finalDistances)[::-1][:k]
      # print ('2) Memory usage: %s (kb)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - previous))
      # previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      # distances = distances[np.argsort(distances[:,0])]
      # neighbours = np.copy(distances[np.argsort(distances[:,0])][0:k])
      # testVectorLabels = np.copy(currentTestVector[4096:])
      # result = np.hstack((currentTestVector[neurons:], neighbourIds))
      # logger.debug("Result shape: " + str(result.shape))
      # logger.debug("Result:")
      # logger.debug(result[::-1][k-1])
      # logger.debug("Results shape: " + str(results.shape))

      results[counter,:] = np.hstack((currentTestVector[neurons:], neighbourIds))
      # results.append({'labelIds': np.array(testVectorLabels, dtype="int8"), 'neighbours': neighbours})
      # logger.info("Calculation for test activation " + str(counter) + " done.")
      counter += 1
      # print ('3) Memory usage: %s (kb)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - previous))
      # previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

      # print ('Overall memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

      # if count % 100 == 0:
      #    print ('Calculated nearest neighbours for ' + str(count) + ' test vectors.')

   logger.info('Calculated ' + str(k) +' nearest neighbours for ' + str(counter) + ' test vectors.')
   return results

def kNearestAnalysed(results, k, labelCount):

   # print results.shape
   # print results[0]['neighbours'].shape
   # print results[0]['labelIds'].shape
   correct = 0
   wrong = 0

   correctDistribution = [0] * labelCount
   wrongDistribution = [0] * labelCount

   for result in results:

      currentDistribution = [0] * labelCount

      count = 0
      kNearest = result['neighbours'][0:k]

      # print kNearest.shape
      # print kNearest[:,1:].shape

      currentDistribution = np.array(kNearest[:,1:], dtype="int16")
      correctLabels = (result['labelIds'] == currentDistribution)

      # print currentDistribution.shape
      # print currentDistribution
      # print result['labelIds'].shape
      # print result['labelIds']
      # print correctLabels.shape
      # print correctLabels

      correctLabelsCount = np.count_nonzero(result['labelIds'])

      predictedWrong = np.count_nonzero((False == correctLabels))

      if (correctLabelsCount * k) - predictedWrong > 0:
         #print "Correct."
         correct += 1
         # correctDistribution[result['labelIds']] += 1
      else:
         #print "Wrong."
         wrong += 1
         # wrongDistribution[result['labelIds']] += 1

   #print 'correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(float(correct)/(wrong + correct))
   #print 'correct per label: ' + str(correctDistribution)
   #print 'wrong per label: ' + str(wrongDistribution)

   return (correct, wrong, correctDistribution, wrongDistribution)
