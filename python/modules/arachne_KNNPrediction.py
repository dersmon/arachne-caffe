import numpy as np
from scipy import sparse

def nearestNeighbours(labeledVectors, newVectors):

	# print np.array(labeledVectors).shape
	# print np.array(newVectors).shape

	results = []
	batchSize = 1000
	count = 0
	for newVector in newVectors:
		distances = np.zeros((len(labeledVectors), (1 + newVector[4096:].size)))
		batchCounter = 0

		# print distances.shape

		while batchCounter < len(labeledVectors):
			# print "new batch"
			currentBatch = np.array(labeledVectors[batchCounter:batchCounter+batchSize])
			# print currentBatch.shape

			currentDifferences = currentBatch[:,0:4096] - newVector[0:4096]
			# print currentDifferences.shape

			currentDistances = np.linalg.norm(currentDifferences, axis=1)
			# print currentDistances.shape
			# print currentDistances.T.shape

			labels =  currentBatch[:,4096:]
			# print labels.shape


			batchResult = np.vstack((currentDistances, labels.T))
			# print batchResult.T.shape

			distances[batchCounter:batchCounter+batchSize,:] = batchResult.T
			batchCounter += batchSize

		# print distances.shape
		# print distances[:,0]
		distances = distances[np.argsort(distances[:,0])]
		# print distances[:,0]
		results.append({'labelIds': np.array(newVector[4096:], dtype="int16"), 'neighbours': distances})

		count += 1

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
