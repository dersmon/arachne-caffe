import numpy as np
from scipy import sparse
import resource
import sys
import types

def nearestNeighbours(labeledVectors, newVectors, k):

	if k == 0:
		k = len(labeledVectors)
	results = []

	batchSize = 1000
	if batchSize > len(labeledVectors):
		batchSize = len(labeledVectors)


	count = 0
	for newVector in newVectors:
		distances = np.zeros((len(labeledVectors), (1 + newVector[4096:].size)))
		batchCounter = 0

		print '1) Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

		while batchCounter < len(labeledVectors):
			# current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			# print "new batch"
			currentBatch = np.array(labeledVectors[batchCounter:batchCounter+batchSize])
			# print currentBatch.shape
			# print '1.1) Memory usage: %s (kb)' % str(current - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			currentBatch[:,0:4096] -= newVector[0:4096]


			# print currentDifferences.shape
			# print '1.2) Memory usage: %s (kb)' % str(current - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			currentDistances = np.linalg.norm(currentBatch[:,0:4096], axis=1)
			# print currentDistances.shape
			# print currentDistances.shape
			# print currentDistances.T.shape
			# print '1.3) Memory usage: %s (kb)' % str(current - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			labels =  currentBatch[:,4096:]
			# print labels.shape
			# print '1.4) Memory usage: %s (kb)' % str(current - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

			batchResult = np.vstack((currentDistances, labels.T))
			# print batchResult.T.shape
			# print '1.5) Memory usage: %s (kb)' % str(current - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			distances[batchCounter:batchCounter+batchSize,:] = batchResult.T
			# print '1.6) Memory usage: %s (kb)' % str(current - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			batchCounter += batchSize

		print '2) Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

		# print distances.shape
		# print distances[:,0]
		distances = distances[np.argsort(distances[:,0])]

		# print distances.shape
		# kDistances = distances[0:k]
		# print kDistances.shape
		# print newVector[4096:].shape
		# print distances[:,0]
		results.append({'labelIds': np.array(newVector[4096:], dtype="int8"), 'neighbours':  distances[np.argsort(distances[:,0])][0:k]})
		# print len(results)
		count += 1

		print '3) Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

		if count % 100 == 0:
			print 'Calculated nearest neighbours for ' + str(count) + ' test vectors.'

	print 'Calculated nearest neighbours for ' + str(count) + ' test vectors.'
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
