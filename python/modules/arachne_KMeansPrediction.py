import numpy as np
import random

def kMeans(activationVectors, labelCount):

	centers = []
	count = 0
	running = 1
	while count < labelCount:
		centers.append({'position':random.choice(activationVectors)[1:], 'clusterMembers': []})
		count += 1

	count = 0
	while count < 100:
		centers = kMeansIteration(centers, activationVectors)
		count += 1

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
			difference = center['position'] - activation[1:]
			distances.append(np.linalg.norm(difference))

		# Assign to closest center
		members = tempCenters[np.argmin(distances)].get('clusterMembers')
		members.append(count)
		tempCenters[np.argmin(distances)]['clusterMembers'] = members

		count += 1


	# Adjust centers towards mean of assigned vectors

	for center in tempCenters:
		#print "Assigned points: " + str(center['clusterMembers'])
		points = [activations[i][1:] for i in center['clusterMembers']]

		updatedPosition = np.sum(points, axis=0)
		if len(center['clusterMembers']) != 0:
			updatedPosition /= len(center['clusterMembers'])

		#print 'old position: ' + str(center['position']) + ', length: ' + str(len(center['position']))
		#print 'new position: ' + str(updatedPosition) + ', length: ' + str(updatedPosition.shape[0])

		updatedCenters.append({'position':updatedPosition, 'clusterMembers':center['clusterMembers']})

	return updatedCenters

def clusterAnalysis(training, clusters, labelCount):

	analysedCluster = []
	counter = 0
	for cluster in clusters:
		print 'Cluster ' + str(counter) + ', labels:'
		points =[int(training[i][0]) for i in cluster['clusterMembers']]

		labelDistribution = [0] * labelCount

		for point in points:
			labelDistribution[point] += 1

		print str(labelDistribution)

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
			difference = center['position'] - activation[1:]
			distances.append(np.linalg.norm(difference))

		centerCounter = 0
		for center in clusters:
			if centerCounter == np.argmin(distances):
				if center['maxLabelID'] == int(activation[0]):
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
