import numpy as np

def nearestNeighbours(labeledVectors, newVectors):
	results = []

	for newVector in newVectors:
		distances = []

		for labeledVector in labeledVectors:
			difference = labeledVector[1:] - newVector[1:]
			distances.append([int(labeledVector[0]), np.linalg.norm(difference)])

		distances = sorted(distances, key=lambda difference: difference[1])

		results.append({'labelId': int(newVector[0]), 'neighbours': distances})

	return results

def kNearestAnalysed(results, k, labelCount):

	correct = 0
	wrong = 0

	correctDistribution = [0] * labelCount
	wrongDistribution = [0] * labelCount

	for result in results:

		currentDistribution = [0] * labelCount

		count = 0
		nNearest = []
		while count < k:
			nNearest.append(result['neighbours'][count])
			count += 1

		for neighbour in nNearest:
			currentDistribution[neighbour[0]] += 1

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
