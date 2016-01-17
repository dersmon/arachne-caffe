import numpy as np
import random
import logging
import time

from multiprocessing import Process, Queue, current_process, freeze_support

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def worker(input, output):
	for func, args in iter(input.get, 'STOP'):
		result = func(*args)
		output.put(result)

def runMultiprocess(self, center, activations):
	distances = []

	# Calculate distance to centers
	differences = center['position'] - activations[:][0:4096].T
	logger.debug(differences.shape)

	differences = np.linalg.norm(differences, None, 1)
	logger.debug('Differences for cluster center ' + __name__)
	logger.debug(differences.shape)

	return differences

def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print "%s processing %s" % (threadName, data)
        else:
            queueLock.release()
        time.sleep(1)

def kMeans(activationVectors, labelCount, maxIterations):

	centers = []
	count = 0
	running = 1

	# Initialize empty cluster centers.
	while count < labelCount:
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

	tempCenters = []
	updatedCenters = []

	for center in centers:
		tempCenters.append({'position':center.get('position'), 'clusterMembers': []})

	count = 0

	for activation in activations:

		distances = []
		# Calculate distance to centers
		for center in centers:
			difference = center['position'] - activation[0:4096]
			distances.append(np.linalg.norm(difference))

		# Assign to closest center
		members = tempCenters[np.argmin(distances)].get('clusterMembers')
		members.append(count)
		tempCenters[np.argmin(distances)]['clusterMembers'] = members

		count += 1

	# Adjust centers towards mean of assigned vectors
	for center in tempCenters:
		# logger.debug("Assigned points: " + str(center['clusterMembers']))
		points = [activations[i][0:4096] for i in center['clusterMembers']]

		updatedPosition = np.sum(points, axis=0)
		if len(center['clusterMembers']) != 0:
			updatedPosition /= len(center['clusterMembers'])

		# logger.debug('old position: ' + str(center['position']) + ', length: ' + str(len(center['position'])))
		# logger.debug('new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0]))

		updatedCenters.append({'position':updatedPosition, 'clusterMembers':center['clusterMembers']})

	return updatedCenters

def clusterAnalysis(training, clusters, labelCount):

	analysedCluster = []
	counter = 0
	for cluster in clusters:
		logger.info('Cluster ' + str(counter) + ', labels:')
		points =[int(training[i][0]) for i in cluster['clusterMembers']]

		labelDistribution = [0] * labelCount

		for point in points:
			labelDistribution[point] += 1

		logger.info(str(labelDistribution))

		maxLabelId = np.argmax(labelDistribution)

		analysedCluster.append({'position': cluster['position'], 'maxLabelID': maxLabelId, 'memberLabelIDs': points, 'labelDistribution': labelDistribution})
		counter += 1

	return analysedCluster

def clusterTest(clusters, testVectors, labelCount):

	tempCenters = []
	updatedCenters = []

	correct = 0
	wrong = 0

	correctPerLabel = [0] * labelCount
	wrongPerLabel = [0] * labelCount

	for activation in testVectors:

		distances = []

		# Calculate distance to centers
		for center in clusters:
			difference = center['position'] - activation[0:4096]
			distances.append(np.linalg.norm(difference))

		centerCounter = 0
		for center in clusters:
			if centerCounter == np.argmin(distances):
				if center['maxLabelID'] == np.argwhere(activation[4096:] == 1)):
					correct += 1
					correctPerLabel[int(activation[0])] += 1
				else:
					wrong += 1
					wrongPerLabel[int(activation[0])] += 1

			centerCounter += 1

	print 'correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(float(correct)/(wrong + correct))
	print 'correct per label: ' + str(correctPerLabel)
	print 'wrong per label: ' + str(wrongPerLabel)

def multipleClustersPerLabel(activationVectors, labelNumber, clusterCount):

	splitActivations = [[]] * labelNumber
	splitCenters = [[]] * labelNumber

	labelCounter = 0
	while labelCounter < labelNumber:
		splitActivations[labelCounter] = []
		splitCenters[labelCounter] = []
		labelCounter += 1

	for vector in activationVectors:
		splitActivations[int(vector[0])].append(vector)

	labelCounter = 0
	while labelCounter < labelNumber:
		centerCounter = 0
		while centerCounter < clusterCount:
			splitCenters[labelCounter].append({'position':random.choice(splitActivations[labelCounter])[1:], 'clusterMembers': []})
			centerCounter += 1
		labelCounter += 1

	labelCounter = 0
	while labelCounter < labelNumber:
		iterations = 0
		while iterations < 100:
			splitCenters[labelCounter] = kMeansIteration(splitCenters[labelCounter], splitActivations[labelCounter])
			iterations += 1
		labelCounter += 1

	return [splitActivations, splitCenters]

def multipleLabelsPerImage(activations, clusterCount, iterations):
	clusters = []

	counter = 0
	while counter < clusterCount:
		clusters.append({'position':random.choice(activations)[0:4096], 'clusterMembers':[]})
		counter += 1

	counter = 0
	while counter < iterations:
		logger.info("KMeans iteration " + str(counter + 1))
		clusters = kMeansIterationMultipleLabels(clusters, activations)
		counter += 1

	return clusters

def kMeansIterationMultipleLabels(clusters, activations):

	tempClusters = []
	updatedClusters = []

	for cluster in clusters:
		tempClusters.append({'position':cluster.get('position'), 'clusterMembers': []})

	count = 0
	for activation in activations:
		distances = []
		# Calculate distance to centers
		for cluster in clusters:
			difference = cluster['position'] - activation[0:4096]
			distances.append(np.linalg.norm(difference))

		# Assign to closest center
		members = tempClusters[np.argmin(distances)].get('clusterMembers')
		members.append(count)
		tempClusters[np.argmin(distances)]['clusterMembers'] = members

		count += 1

	# Adjust centers towards mean of assigned vectors

	for cluster in tempClusters:
		#print "Assigned points: " + str(center['clusterMembers'])
		points = [activations[i][0:4096] for i in cluster['clusterMembers']]

		updatedPosition = np.sum(points, axis=0)
		if len(cluster['clusterMembers']) != 0:
			updatedPosition /= len(cluster['clusterMembers'])

		#print 'old position: ' + str(center['position']) + ', length: ' + str(len(center['position']))
		#print 'new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0])

		updatedClusters.append({'position':updatedPosition, 'clusterMembers': cluster['clusterMembers']})

	return updatedClusters

def clusterAnalysisMultipleLabels(clusters, training):

	analysedCluster = []
	counter = 0
	for cluster in clusters:
		print 'Cluster ' + str(counter) + ', labels:'
		labels = [np.array(training[i,4096:], dtype='int16') for i in cluster['clusterMembers']]

		labelDistribution = [0] * len(labels[0])

		for label in labels:
			labelDistribution += label

		print str(labelDistribution)

		maxLabelId = np.argmax(labelDistribution)

		analysedCluster.append({'position': cluster['position'], 'maxLabelID': maxLabelId, 'memberLabelIDs': labels, 'labelDistribution': labelDistribution})
		counter += 1

	return analysedCluster
