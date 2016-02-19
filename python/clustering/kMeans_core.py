import numpy as np
import random
import logging
import pickle

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def runKMeans(activations, labelCount, clusterCount, maxIterations):

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
   while iteration < maxIterations:
      # logger.info("Running KMeans iteration " + str(iteration) + ".")
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

   filteredClusters = [] # filter cluster without members at the end
   for cluster in clusters:
      if(len(cluster['memberIndices']) != 0):
         filteredClusters.append(cluster)


   return [filteredClusters, iteration]

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
      memberLabelHistogram = np.sum(memberLabelHistogram, axis=0)


      updatedPosition = np.sum(points, axis=0)
      if len(membersIndices) != 0:
         updatedPosition /= len(membersIndices)

      logger.debug('Points shape: ' + str(np.array(points).shape))
      logger.debug('Label histogram shape: ' + str(memberLabelHistogram.shape))
      logger.debug('Updated position shape: ' + str(updatedPosition.shape))
      if membersIndices.size == 0: # this cluster has no members assigned, create empty members and keep old position
         logger.info('Cluster without members.')
         memberLabelHistogram = np.zeros((activations.shape[1] - firstLabelIndex))
         logger.info('Histogram: ' + str(memberLabelHistogram))
         membersIndices = []
         updatedPosition = cluster['position']

      updatedClusters.append({'position': updatedPosition, 'memberLabelHistogram': memberLabelHistogram,  'memberIndices':membersIndices})
      counter += 1

   return updatedClusters

def cleanUp(clusters):
   logger.debug("Cluster count: " + str(len(clusters)))
   data = []
   for cluster in clusters:
      # logger.debug("Position: " + str(cluster['position'].shape))
      # logger.debug("Histogram: " + str(cluster['memberLabelHistogram'].shape))
      data.append(np.hstack((cluster['position'], cluster['memberLabelHistogram'])))
   return np.array(data)

def saveResults(clusters, iterations, targetPath):
   with open(targetPath + 'clusters_with_members.pickle', "w") as outputFile:
      pickle.dump(clusters, outputFile)
   data = cleanUp(clusters)
   with open(targetPath + 'clusters.npy', "w") as outputFile:
      np.save(outputFile, data)
   with open(targetPath + 'iterations.npy', "w") as outputFile:
      np.save(outputFile, iterations)

def predictSimple(clusters, activation, neurons):
   distances = [[] for i in range(len(clusters))]

   clusterCounter = 0

   for cluster in clusters:
      difference = activation[0:neurons] - cluster[0:neurons]
      distances[clusterCounter] = np.linalg.norm(difference)
      clusterCounter += 1

   clusterRanking = np.argsort(distances)

   return clusterRanking

def predictBayes(clusters, activations, labels, targetPath):
   logger.info('Todo.')
