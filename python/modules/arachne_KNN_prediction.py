import numpy as np
from scipy import sparse
import resource
import sys
import types

def getKNearestNeighbours(training, test, k):

	if k == 0:
		k = len(training)
	results = []

	batchSize = 1000
	if batchSize > len(training):
		batchSize = len(training)

	print "Searching " + str(k) + " nearest neighbours. Batch size is " + str(batchSize) + "."
	previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

	count = 0
	for currentTestVector in test:
		distances = np.empty((len(training), (1 + currentTestVector[4096:].size)), dtype='float16')
		batchCounter = 0

		print ('1) Memory usage: %s (kb)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - previous))
		previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

		while batchCounter < len(training):

			end = batchCounter + batchSize
			if end > len(training):
				end = len(training)

			currentBatch = np.array(training[batchCounter:end])
			currentBatch[:,0:4096] -= currentTestVector[0:4096]

			batchResult = np.vstack((np.linalg.norm(currentBatch[:,0:4096], axis=1), currentBatch[:,4096:].T))

			distances[batchCounter:batchCounter+batchSize,:] = batchResult.T

			batchCounter += batchSize

		print ('2) Memory usage: %s (kb)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - previous))
		previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		distances = distances[np.argsort(distances[:,0])]
		neighbours = np.copy(distances[np.argsort(distances[:,0])][0:k])
		testVectorLabels = np.copy(currentTestVector[4096:])


		results.append({'labelIds': np.array(testVectorLabels, dtype="int8"), 'neighbours': neighbours})

		count += 1
		print ('3) Memory usage: %s (kb)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - previous))
		previous = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

		print ('Overall memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

		if count % 100 == 0:
			print ('Calculated nearest neighbours for ' + str(count) + ' test vectors.')

	print ('Calculated nearest neighbours for ' + str(count) + ' test vectors.')
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
