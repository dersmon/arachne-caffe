import sys
import os
import logging

import numpy as np

import modules.utility as utility
import clustering.kNearestNeighbours as knn

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAX_K = 3

def runTests(training, test, labels, targetPath, nn):
   global MAX_K

   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))
   labelCount = len(labels)
   MAX_K = training.shape[0] / labelCount
   neurons = training.shape[1] - labelCount
   if nn == None:
      nn = knn.getKNearestNeighbours(training, test, MAX_K, neurons)
      with open(targetPath + "nn.npy", "w") as outputFile:
         np.save(outputFile, nn)

   nnByLabel = [[] for i in range(labelCount)]
   currentLabelIndex = 0
   while currentLabelIndex < labelCount:
      currentSelection = nn[nn[:, currentLabelIndex] == 1]
      nnByLabel[currentLabelIndex] = currentSelection
      currentLabelIndex += 1

   allLabels = training[:,neurons:]

   sumPerLabel = np.sum(allLabels, axis=0)
   factorPerLabel = np.max(sumPerLabel) / sumPerLabel


   kCounter = 1
   results = []
   while kCounter < MAX_K:
      logger.info("Evaluating K = " + str(kCounter) + ":")
      overallCorrect = 0
      overallWrong = 0
      meanAveragePrecision = 0
      confusionMatrix = np.zeros((labelCount,labelCount))

      for labelIndex, values in enumerate(nnByLabel):
         confusion = np.zeros((1,len(labels)))
         activationCounter = 0
         while activationCounter < values.shape[0]:
            currentActivation = values[activationCounter]
            searchedLabel = np.argwhere(currentActivation[:labelCount] == 1)
            nIndices = currentActivation[labelCount:labelCount+kCounter].astype(int)
            neighbourLabels = training[nIndices][:,neurons:]

            neighbourLabelsSum = np.sum(neighbourLabels, axis=0)
            neighbourLabelsSum = neighbourLabelsSum * factorPerLabel

            sortedNeighbourLabels = np.argsort(neighbourLabelsSum)[::-1]

            if sortedNeighbourLabels[0] == searchedLabel:
               overallCorrect += 1
            else:
               overallWrong += 1

            confusion[0, sortedNeighbourLabels[0]] += 1

            averagePrecision = 0
            relevant = 0

            for idx, value in enumerate(sortedNeighbourLabels):
               indicator = 0
               if(value == searchedLabel and neighbourLabelsSum[value] != 0):
                  indicator = 1
                  relevant += 1

               precision = float(relevant) / (idx + 1)
               averagePrecision += (precision * indicator)

            if relevant != 0:
               averagePrecision = float(averagePrecision) / relevant
            meanAveragePrecision += averagePrecision
            activationCounter += 1

         confusionMatrix[labelIndex,:] = confusion

      meanAveragePrecision = float(meanAveragePrecision) / test.shape[0]
      logger.info(' Accuracy: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))
      logger.info(' Mean average precision: '+str(meanAveragePrecision))

      currentTargetPath = targetPath + str(kCounter) + "/"
      if not os.path.exists(os.path.dirname(currentTargetPath)):
         os.makedirs(os.path.dirname(currentTargetPath))

      logger.info('Saving confusion matrix ' + currentTargetPath + "confusion.npy")
      with open(currentTargetPath + "confusion.npy", "w") as outputFile:
         np.save(outputFile, confusionMatrix)

      results.append([kCounter, meanAveragePrecision, overallCorrect, overallWrong])
      utility.plotConfusionMatrix(confusionMatrix, labels, currentTargetPath + "confusion.pdf")

      kCounter += 1

   results = np.array(results)

   logger.info('Writing file ' + targetPath + "overview.csv")
   np.savetxt( targetPath + "overview.csv", results, delimiter=',')

   utility.plotKMeansOverview(results, targetPath + "overview.pdf", False)


if __name__ == '__main__':
   if len(sys.argv) != 5 and len(sys.argv) != 6:
      logger.info("Please provide as argument:")
      logger.info("1) Path to training activations (*.npy).")
      logger.info("2) Path to test activations (*.npy).")
      logger.info("3) Path to label mapping.")
      logger.info("4) Path for target folder.")
      logger.info("5) Neighbour relations (optional)")
      sys.exit();

   trainingActivations = utility.arrayFromFile(sys.argv[1])
   testActivations = utility.arrayFromFile(sys.argv[2])
   labels = utility.getLabelStrings(sys.argv[3])
   nn = None
   if len(sys.argv) == 6:
      nn = utility.arrayFromFile(sys.argv[5])

   runTests(trainingActivations, testActivations, labels, sys.argv[4], nn)
