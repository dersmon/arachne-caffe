import sys
import os
import numpy as np

import modules.utility as utility
import logging

import train_bayes as tb

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def testBayes(training, test, labels, targetPath):

   if targetPath.endswith('/') == False:
      targetPath += '/'
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   labelCount = len(labels)
   neurons = training.shape[1] - labelCount

   minTraining = np.min(training[:,0:neurons])
   minTest = np.min(test[:,0:neurons])
   absoluteMin = 0

   if minTraining < minTest:
      absoluteMin = minTraining
   else:
      absoluteMin = minTest

   training[:,0:neurons] = training[:,0:neurons] + np.abs(absoluteMin)
   test[:,0:neurons] = test[:,0:neurons] + np.abs(absoluteMin)

   PLog = tb.trainBayes(training, labelCount)

   overallCorrect = 0
   overallWrong = 0
   meanAveragePrecision = 0

   featuresByLabel = utility.splitTestFeaturesByLabel(test, len(labels))

   confusionMatrix = np.zeros((labelCount,labelCount))

   for labelIndex, activations in enumerate(featuresByLabel):
      counter = 0
      while counter < activations.shape[0]:
         currentActivation = activations[counter,0:neurons]
         searchedLabel = np.argmax(activations[counter,neurons:])

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

   meanAveragePrecision = float(meanAveragePrecision) / test.shape[0]

   logger.info(' Accuracy: ' + str(float(overallCorrect)/(overallWrong + overallCorrect)))
   logger.info(' Mean average precision: '+str(meanAveragePrecision))

   utility.plotConfusionMatrix(confusionMatrix, labels, targetPath + "confusion.pdf")

   results = [0, meanAveragePrecision, overallCorrect, overallWrong]
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
