import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def getLabelStrings(filePath):
   with open(filePath, 'r') as inputFile:
      result = []
      for line in inputFile.readlines():
        result.append(line.split(' ')[0])

   return result

def arrayFromFile(filePath):
   logger.info('Opening file: ' + filePath)
   with open(filePath, 'r') as inputFile:
      return np.load(inputFile)

def plotConfusionMatrix(confusionMatrix, labels, evaluationTargetPath):
   maxValue = np.max(confusionMatrix, axis=1)

   ax = plt.gca()
   scaled = (confusionMatrix.T / maxValue).T

   for(j,i),label in np.ndenumerate(confusionMatrix):
      ax.text(i,j, int(label), ha='center', va='center', color='black',  fontsize=8)

   ticks = np.arange(len(labels))

   ax.set_yticks(ticks)
   ax.set_yticklabels(labels)
   ax.set_xticks(ticks)
   ax.set_xticklabels(labels, rotation=45, ha='right')

   plt.imshow(scaled, 'Blues', interpolation='none')
   plt.savefig(evaluationTargetPath + 'confusionMatrix.pdf', bbox_inches='tight')
