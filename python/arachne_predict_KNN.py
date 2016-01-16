import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import modules.arachne_caffe as ac
import modules.arachne_KNN_prediction as aknn
import modules.arachne_plotting as ap

def testKNN(training, test, labelCount, k):
   filePath = "./neighbours_elastic.npy"

   if k == 0:
      k = len(training)

   testNeighbours = aknn.getKNearestNeighbours(training, test, k)

   print 'Writing file ' + filePath
   if not os.path.exists(os.path.dirname(filePath)):
   	os.makedirs(os.path.dirname(filePath))
   with open(filePath, "w") as outputFile:
   	np.save(outputFile, testNeighbours)

   # with open(filePath, 'r') as inputFile:
   # 	testNeighbours = np.load(inputFile)

   count = 0
   data = []
   while count < k and count < len(training):

   	(correct, wrong, correctPerLabel, wrongPerLabel) = aknn.kNearestAnalysed(testNeighbours, count + 1, labelCount)

   	data.append([count, float(correct)/(wrong + correct)])
   	count += 1
   	if((count + 1) % 100 == 0):
   		print 'Calculated prediction for ' + str(count + 1) + ' nearest neighbours.'

   	print str(correct) + ", " + str(wrong) + ", " + str(float(correct)/(wrong + correct))

   print 'Calculated prediction for ' + str(count) + ' nearest neighbours.'
   data = np.array(data)

   plt.plot(data[:,0], data[:,1], 'k')
   plt.axis([1, len(data), 0, 1])
   plt.grid(True)
   plt.show()

trainingActivationsPath = ""
testActivationsPath = ""

trainingActivations = None
testActivations = None

if(len(sys.argv) < 2):
	print("No activation vectors provided.")
else:
	trainingActivationsPath = sys.argv[1]

if(len(sys.argv) < 3):
	print("No activation vectors provided.")
else:
	testActivationsPath = sys.argv[2]

if trainingActivationsPath.endswith('.npy'):
	trainingActivations = ac.activationsFromFile(trainingActivationsPath)
else:
	print(trainingActivationsPath + " does not seem to be a npy-file with activations.")

if testActivationsPath.endswith('.npy'):
	testActivations = ac.activationsFromFile(testActivationsPath)
else:
	print(testActivations + " does not seem to be a npy-file with activations.")

labelCount = len(trainingActivations[0][4096:])

# ap.plotActivations(trainingActivations, 4096)
# ap.plotActivations(testActivations, 4096)


testKNN(trainingActivations, testActivations, labelCount, 50)
