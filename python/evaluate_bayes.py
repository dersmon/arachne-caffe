import sys
import os
import numpy as np

import modules.utility as utility
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def createClassifier(training, labelCount):

   neurons = training.shape[1] - labelCount

   activationsByLabel = []
   counter = 0
   while counter < labelCount:
      currentLabelIndex = training.shape[1] - labelCount + counter
      currentSelection = training[training[:, currentLabelIndex] == 1][:,0:neurons]
      currentSelection = np.sum(currentSelection, axis=0)
      activationsByLabel.append(currentSelection)
      counter += 1

   activationsByLabel = np.array(activationsByLabel)
   devisor = np.tile(np.sum(activationsByLabel, axis=1),(neurons, 1)).T
   P = (activationsByLabel + 1) / (devisor + neurons)

   return P

def testBayes(training, test, labels, targetPath):

   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   labelCount = len(labels)

   P = createClassifier(training, labelCount)
   neurons = training.shape[1] - labelCount

   overallCorrect = 0
   overallWrong = 0
   meanAveragePrecision = 0

   activationsByLabel = [[] for i in range(labelCount)]
   counter = 0
   while counter < labelCount:
      currentLabelIndex = test.shape[1] - labelCount + counter
      currentSelection = test[test[:, currentLabelIndex] == 1]
      activationsByLabel[counter] = currentSelection
      counter += 1

   confusionMatrix = np.zeros((labelCount,labelCount))
   for labelIndex, activations in enumerate(activationsByLabel):
      counter = 0
      while counter < activations.shape[0]:
         currentActivation = activations[counter,0:neurons]
         searchedLabel = np.argmax(activations[counter,neurons:])
         PLog = np.log(P)

         predictions = currentActivation * PLog
         predictions = np.sum(predictions, axis=1)

         if np.argmax(predictions) == searchedLabel:
            overallCorrect += 1
         else:
            overallWrong += 1

         confusionMatrix[labelIndex,np.argmax(predictions)] += 1

         averagePrecision = 0
         relevant = 0

         predictedLabelsSorted = np.argsort(predictions)[::-1]

         for idx, value in enumerate(predictedLabelsSorted):
            indicator = 0
            if(value == searchedLabel):
               indicator = 1
               relevant += 1

            precision = float(relevant) / (idx + 1)
            averagePrecision += (precision * indicator)

         if relevant != 0:
            averagePrecision = float(averagePrecision) / relevant
         meanAveragePrecision += averagePrecision

         counter += 1

   logger.info('Accuracy Bayes:')
   logger.info('correct: ' + str(overallCorrect) + ', wrong: ' + str(overallWrong) + ', ratio: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))
   meanAveragePrecision = float(meanAveragePrecision) / test.shape[0]
   logger.info(' Mean average precision: '+str(meanAveragePrecision))

   utility.plotConfusionMatrix(confusionMatrix, labels, targetPath + "confusion.pdf")

   results = [[meanAveragePrecision, overallCorrect, overallWrong]]
   logger.info('Writing file ' + targetPath + "overview.csv")
   np.savetxt( targetPath + "overview.csv", results, delimiter=',')

if __name__ == '__main__':
   if len(sys.argv) != 5:
      logger.info("Please provide as argument:")
      logger.info("1) Path to training activations (*.npy).")
      logger.info("2) Path to test activations (*.npy).")
      logger.info("3) Path to label mapping.")
      logger.info("4) Path for target folder.")
      sys.exit();

   trainingActivations = utility.arrayFromFile(sys.argv[1])
   testActivations = utility.arrayFromFile(sys.argv[2])

   labels = utility.getLabelStrings(sys.argv[3])
   targetPath = sys.argv[4]

   if targetPath.endswith("/") == False:
      targetPath += "/"

   testBayes(trainingActivations, testActivations, labels, targetPath)
