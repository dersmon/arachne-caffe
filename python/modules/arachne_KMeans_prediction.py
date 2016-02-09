import numpy as np
import random
import logging
import time

from multiprocessing import Process, Queue, current_process, freeze_support

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def kMeans(activationVectors, clusterCount, maxIterations):

	centers = []
	count = 0
	running = 1

	# Initialize empty cluster centers.
	while count < clusterCount:
		centers.append({'position':random.choice(activationVectors)[0:4096], 'clusterMembers': []})
		count += 1

	# Run K-means iterations.
	iteration = 0
	while iteration < maxIterations:
		logger.info("Running KMeans iteration " + str(iteration) + ".")
		oldCenters = centers
		centers = kMeansIteration(centers, activationVectors)

		changed = False
		for idx, val in enumerate(oldCenters):
			if not (oldCenters[idx]['position'] == centers[idx]['position']).all():
				changed = True

		if not (changed):
			logger.info("Cluster centers did not change positions. Iteration " + str(iteration))
			break

		iteration += 1

	return centers

def kMeansIteration(centers, activations):

	# tempCenters = []
	updatedCenters = []
	#
	# for center in centers:
	# 	tempCenters.append({'position':center.get('position'), 'clusterMembers': []})
	# logger.debug(len(activations))
	# logger.debug(activations.shape)
	count = 0

	# logger.debug(len(activations))


	distances = []
	for center in centers:
		# logger.debug("New center: ")
		# logger.debug(center['position'].shape)
		# logger.debug(activations[:,0:4096].shape)
		centerDistances = []
		batchCount = 0
		batchSize = 10000

		while(batchCount < len(activations)):
			differences = (activations[batchCount:(batchCount+batchSize),0:4096] - center['position']).T
			# logger.debug(differences.shape)
			temp = np.linalg.norm(differences, None, 0)
			# logger.debug(temp.shape)
			centerDistances.extend(np.linalg.norm(differences, None, 0))
			batchCount += batchSize

		# logger.debug(len(centerDistances))

		distances.append(np.array(centerDistances))
		# distances for each activation (1:41xx o.ae.)

	distances = np.array(distances)

	# logger.debug("Complete: ")
	# logger.debug(len(distances))
	# logger.debug(distances[0].shape)

	minDistanceIndex = np.argmin(distances, axis=0)
	# logger.debug(minDistanceIndex)
	# logger.debug(np.unique(minDistanceIndex))
	# (4105,) = minDistanceIndex.shape, each value is index of cluster nearest to value

	counter = 0
	for center in centers:
		members = np.where(minDistanceIndex == counter)[0]
		points = [activations[i,0:4096] for i in members]

		# logger.debug("Members:")
		# logger.debug(members)
		# logger.debug(members.shape)
		# logger.debug("Coordinates:")
		# logger.debug(np.array(points).shape)

		updatedPosition = np.sum(points, axis=0)
		if len(members) != 0:
			updatedPosition /= len(members)

		updatedCenters.append({'position':updatedPosition, 'clusterMembers':members})
		counter += 1


	# for activation in activations:
	#
	# 	distances = []
	# 	# Calculate distance to centers
	# 	for center in centers:
	# 		difference = center['position'] - activation[0:4096]
	# 		distances.append(np.linalg.norm(difference))
	#
	# 	# Assign to closest center
	# 	members = tempCenters[np.argmin(distances)].get('clusterMembers')
	# 	members.append(count)
	# 	tempCenters[np.argmin(distances)]['clusterMembers'] = members
	#
	# 	count += 1

	# Adjust centers towards mean of assigned vectors
	# for center in tempCenters:
		# logger.debug("Assigned points: " + str(center['clusterMembers']))
		# points = [activations[i][0:4096] for i in center['clusterMembers']]
		#
		# updatedPosition = np.sum(points, axis=0)
		# if len(center['clusterMembers']) != 0:
		# 	updatedPosition /= len(center['clusterMembers'])

		# logger.debug('old position: ' + str(center['position']) + ', length: ' + str(len(center['position'])))
		# logger.debug('new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0]))

		# updatedCenters.append({'position':updatedPosition, 'clusterMembers':center['clusterMembers']})

	return updatedCenters

def clusterAnalysis(clusters, training, labelCount):
   analysedCluster = []
   counter = 0
   for cluster in clusters:
      logger.info('Cluster ' + str(counter) + ', labels:')
      labelFlags = [training[i,4096:] for i in cluster['clusterMembers']]
      maxLabelId = -1
      labelDistribution = [0] * labelCount
      logger.debug(labelFlags)
      labelIndices = np.where(np.array(labelFlags[:]) == 1)
      if len(labelFlags) != 0:
         logger.debug(labelIndices[1])
         for index in labelIndices[1]:
            labelDistribution[index] += 1

         logger.info(labelDistribution)
         maxLabelId = np.argmax(labelDistribution)

      analysedCluster.append({'position': cluster['position'], 'maxLabelID': maxLabelId, 'memberLabelIDs': labelIndices, 'labelDistribution': labelDistribution})
      counter += 1

   return analysedCluster

def clusterTest(clusters, testVectors):

   labelCount = len(testVectors[0][4096:])
   tempCenters = []
   updatedCenters = []

   correct = 0
   wrong = 0

   correctRanked = 0
   wrongRanked = 0

   correctPerLabel = [0] * labelCount
   wrongPerLabel = [0] * labelCount

   meanAveragePrecision = 0
   for activation in testVectors:
      distances = []

      # Calculate distance to centers
      for center in clusters:
         difference = center['position'] - activation[0:4096]
         distances.append(np.linalg.norm(difference))

         # logger.debug(distances)
         # logger.debug(np.argsort(distances))

      sortedIndices = np.argsort(distances)

      centerCounter = 0
      if clusters[sortedIndices[0]]['maxLabelID'] == np.argwhere(activation[4096:] == 1):
         correct += 1
         correctPerLabel[np.argwhere(activation[4096:] == 1)] += 1
      else:
         wrong += 1
         wrongPerLabel[np.argwhere(activation[4096:] == 1)] += 1

      maxLabel = 0
      sumCorrectLabel = 0

      # logger.debug(clusters[sortedIndices[0]]['labelDistribution'])


      for idx, labelCount in enumerate(clusters[sortedIndices[0]]['labelDistribution']):
         if labelCount > maxLabel:
            maxLabel = labelCount
         if(idx == np.argwhere(activation[4096:] == 1)):
            sumCorrectLabel += labelCount

      # Mean average precision?
      # precision: |relevant of retreived| / |retreived|
      # average precision: k=1 => n (precision with k first * 0 (=wrong) or 1(=correct))/relevant
      # sum all average precision / querycount

      averagePrecision = 0
      relevant = 0

      # logger.debug(clusters[sortedIndices[0]]['labelDistribution'])
      sortedHistogramIndices = np.argsort(clusters[sortedIndices[0]]['labelDistribution'])[::-1].tolist()
      # logger.debug(sortedHistogramIndices)
      for idx, value in enumerate(sortedHistogramIndices):
         # logger.debug(idx)
         # logger.debug(labelCount)
         indicator = 0
         if(value == np.argwhere(activation[4096:] == 1)):
            indicator = 1
            relevant += 1

         precision = float(relevant) / (idx + 1)
         averagePrecision += (precision * indicator)

         # logger.debug('relevant: ' + str(indicator))
         # logger.debug('indicator: ' + str(indicator))
         # logger.debug('precision: ' + str(averagePrecision))

      averagePrecision = float(averagePrecision) / relevant
      # logger.debug('average precision: ' + str(averagePrecision))

      meanAveragePrecision += averagePrecision
      correctRanked += (float(sumCorrectLabel) / maxLabel)


   logger.info('Exact prediction:')
   logger.info('correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(float(correct)/(wrong + correct)))
   logger.info('correct per label: ' + str(correctPerLabel))
   logger.info('wrong per label: ' + str(wrongPerLabel))

   logger.info('Correct ranked: ')
   logger.info(correctRanked / len(testVectors))

   meanAveragePrecision = float(meanAveragePrecision) / testVectors.shape[0]
   logger.info('Mean average precision:')
   logger.info(meanAveragePrecision)

#
# def multipleClustersPerLabel(activationVectors, labelNumber, clusterCount):
#
# 	splitActivations = [[]] * labelNumber
# 	splitCenters = [[]] * labelNumber
#
# 	labelCounter = 0
# 	while labelCounter < labelNumber:
# 		splitActivations[labelCounter] = []
# 		splitCenters[labelCounter] = []
# 		labelCounter += 1
#
# 	for vector in activationVectors:
# 		splitActivations[int(vector[0])].append(vector)
#
# 	labelCounter = 0
# 	while labelCounter < labelNumber:
# 		centerCounter = 0
# 		while centerCounter < clusterCount:
# 			splitCenters[labelCounter].append({'position':random.choice(splitActivations[labelCounter])[1:], 'clusterMembers': []})
# 			centerCounter += 1
# 		labelCounter += 1
#
# 	labelCounter = 0
# 	while labelCounter < labelNumber:
# 		iterations = 0
# 		while iterations < 100:
# 			splitCenters[labelCounter] = kMeansIteration(splitCenters[labelCounter], splitActivations[labelCounter])
# 			iterations += 1
# 		labelCounter += 1
#
# 	return [splitActivations, splitCenters]
#
# def multipleLabelsPerImage(activations, clusterCount, iterations):
# 	clusters = []
#
# 	counter = 0
# 	while counter < clusterCount:
# 		clusters.append({'position':random.choice(activations)[0:4096], 'clusterMembers':[]})
# 		counter += 1
#
# 	counter = 0
# 	while counter < iterations:
# 		logger.info("KMeans iteration " + str(counter + 1))
# 		clusters = kMeansIterationMultipleLabels(clusters, activations)
# 		counter += 1
#
# 	return clusters
#
# def kMeansIterationMultipleLabels(clusters, activations):
#
# 	tempClusters = []
# 	updatedClusters = []
#
# 	for cluster in clusters:
# 		tempClusters.append({'position':cluster.get('position'), 'clusterMembers': []})
#
# 	count = 0
# 	for activation in activations:
# 		distances = []
# 		# Calculate distance to centers
# 		for cluster in clusters:
# 			difference = cluster['position'] - activation[0:4096]
# 			distances.append(np.linalg.norm(difference))
#
# 		# Assign to closest center
# 		members = tempClusters[np.argmin(distances)].get('clusterMembers')
# 		members.append(count)
# 		tempClusters[np.argmin(distances)]['clusterMembers'] = members
#
# 		count += 1
#
# 	# Adjust centers towards mean of assigned vectors
#
# 	for cluster in tempClusters:
# 		#print "Assigned points: " + str(center['clusterMembers'])
# 		points = [activations[i][0:4096] for i in cluster['clusterMembers']]
#
# 		updatedPosition = np.sum(points, axis=0)
# 		if len(cluster['clusterMembers']) != 0:
# 			updatedPosition /= len(cluster['clusterMembers'])
#
# 		#print 'old position: ' + str(center['position']) + ', length: ' + str(len(center['position']))
# 		#print 'new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0])
#
# 		updatedClusters.append({'position':updatedPosition, 'clusterMembers': cluster['clusterMembers']})
#
# 	return updatedClusters
#
# def clusterAnalysisMultipleLabels(clusters, training):
#
# 	analysedCluster = []
# 	counter = 0
# 	for cluster in clusters:
# 		print 'Cluster ' + str(counter) + ', labels:'
# 		labels = [np.array(training[i,4096:], dtype='int16') for i in cluster['clusterMembers']]
#
# 		labelDistribution = [0] * len(labels[0])
#
# 		for label in labels:
# 			labelDistribution += label
#
# 		print str(labelDistribution)
#
# 		maxLabelId = np.argmax(labelDistribution)
#
# 		analysedCluster.append({'position': cluster['position'], 'maxLabelID': maxLabelId, 'memberLabelIDs': labels, 'labelDistribution': labelDistribution})
# 		counter += 1
#
# 	return analysedCluster
