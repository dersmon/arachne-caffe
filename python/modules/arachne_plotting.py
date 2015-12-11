import matplotlib.pyplot as plt
import numpy as np


def plotActivations(activations, labelStartIndex):

	labels = np.array(activations)[:,labelStartIndex:]
	grid = np.array(activations)[:,0:labelStartIndex]

	maxValue = np.amax(grid)
	minValue = np.amin(grid)

	# print 'max activation: ' + str(maxValue) + ', min activation: ' + str(minValue)

	scaled = grid * (255 / maxValue)

	plt.imshow(scaled, 'Greys_r', interpolation='none')
	plt.show()
