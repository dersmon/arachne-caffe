import numpy as np
import random
import logging
import sys
import os

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAX_ITERATIONS = 150

def runKMeans(activations, labelCount, clusterCount):

   firstLabelIndex = activations.shape[1] - labelCount
   logger.debug('First label index: ' + str(firstLabelIndex))

   clusters = []

   # Initialize empty cluster clusters.
   counter = 0
   while counter < clusterCount:
      current = {'position':random.choice(activations)[0:firstLabelIndex], 'memberLabelHistogram':np.zeros((1, labelCount)), 'memberIndices': []}
      clusters.append(current)
      counter += 1

   # Run K-means iterations.
   iteration = 0
   while iteration < MAX_ITERATIONS:
      logger.info("Running KMeans iteration " + str(iteration) + ".")
      previousclusters = clusters
      clusters = kMeansIteration(clusters, activations, firstLabelIndex)
      changed = False
      for idx, val in enumerate(previousclusters):
         if (previousclusters[idx]['position'] == clusters[idx]['position']).all() == False:
            changed = True

      if changed == False:
         logger.info("Cluster centers did not change positions. Iteration " + str(iteration))
         break

      iteration += 1

   return clusters

def kMeansIteration(clusters, activations, firstLabelIndex):

   updatedClusters = []
   count = 0

   distances = []
   for cluster in clusters:
      clusterDistances = []
      batchCount = 0
      batchSize = 10000

      while(batchCount < len(activations)):
         differences = (activations[batchCount:(batchCount+batchSize),0:firstLabelIndex] - cluster['position']).T
         squaredDistance = np.linalg.norm(differences, None, 0)
         clusterDistances.extend(np.linalg.norm(differences, None, 0))
         batchCount += batchSize

      distances.append(np.array(clusterDistances))

   distances = np.array(distances)

   minDistanceIndex = np.argmin(distances, axis=0)

   counter = 0
   for cluster in clusters:
      membersIndices = np.where(minDistanceIndex == counter)[0]

      points = [activations[i,0:firstLabelIndex] for i in membersIndices]
      memberLabelHistogram = np.array([activations[i,firstLabelIndex:] for i in membersIndices])
      memberLabelHistogram = np.sum(labelHistogram, axis=0)


      updatedPosition = np.sum(points, axis=0)
      if len(members) != 0:
         updatedPosition /= len(members)

      logger.debug('Points shape: ' + str(np.array(points).shape))
      logger.debug('Label histogram shape: ' + str(memberLabelHistogram.shape))
      logger.debug('Updated position shape: ' + str(updatedPosition.shape))
      if membersIndices.shape == False: # this cluster has no members assigned yet, create empty members and keep position
         memberLabelHistogram = np.zeros((1, firstLabelIndex))
         membersIndices = []
         updatedPosition = cluster['position']

      updatedclusters.append({'position': updatedPosition, 'memberLabelHistogram': memberLabelHistogram,  'memberIndices':membersIndices})
      counter += 1

   return updatedclusters

if __name__ == '__main__':

   if len(sys.argv) != 4 and len(sys.argv) != 5:
      logger.info("Please provide as argument:")
      logger.info("1) npy-file with activations.")
      logger.info("2) k.")
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

   clusters = runKMeans(activations, labelCount, k)

   with open(targetPath + 'mixedClusters.npy', "w") as outputFile:
      np.save(outputFile, clusters)

   # split activations by label
   activationsByLabel = []
