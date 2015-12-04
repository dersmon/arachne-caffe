import numpy as np


def nearestNeighbours(labeledVectors, newVectors):
	results = []
	batchSize = 1000
	count = 0
	for newVector in newVectors:
		distances = np.zeros([2,len(labeledVectors)])
		batchCounter = 0

		while batchCounter < len(labeledVectors):
			currentBatch = labeledVectors[batchCounter:batchCounter+batchSize]
			currentBatch = np.array(currentBatch)

			currentDifferences = currentBatch[:,1:] - newVector[1:]
			currentDistances = np.linalg.norm(currentDifferences, axis=1)

			batchResult = np.vstack((currentBatch[:,0], currentDistances))
			distances[:,batchCounter:batchCounter+batchSize] = batchResult
			batchCounter += batchSize

		distances = distances[:,np.argsort(distances[1])]
		results.append({'labelId': int(newVector[0]), 'neighbours': distances})

		count += 1

		if count % 100 == 0:
			print 'Calculated nearest neighbours for ' + str(count) + ' test vectors.'

	print 'Calculated nearest neighbours for ' + str(count) + ' test vectors.'
	return results

def kNearestAnalysed(results, k, labelCount):

	correct = 0
	wrong = 0

	correctDistribution = [0] * labelCount
	wrongDistribution = [0] * labelCount

	for result in results:

		currentDistribution = [0] * labelCount

		count = 0
		kNearest = result['neighbours'][:,0:k]
		for value in kNearest[0,:].tolist():
			currentDistribution[int(value)] += 1

		if np.argmax(currentDistribution) == result['labelId']:
			correct += 1
			correctDistribution[result['labelId']] += 1
		else:
			wrong += 1
			wrongDistribution[result['labelId']] += 1

	#print 'correct: ' + str(correct) + ', wrong: ' + str(wrong) + ', ratio: ' + str(float(correct)/(wrong + correct))
	#print 'correct per label: ' + str(correctDistribution)
	#print 'wrong per label: ' + str(wrongDistribution)

	return (correct, wrong, correctDistribution, wrongDistribution)
