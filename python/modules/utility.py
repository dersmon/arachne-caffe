import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def getLabelStrings(filePath):
   with open(filePath, 'r') as inputFile:
      result = []
      for line in inputFile.readlines():
         if '\t' in line:
            result.append(line.split('\t')[0])
         else:
            result.append(line.split(' ')[0])

   return result

def arrayFromFile(filePath):
   logger.info('Opening file: ' + filePath)
   with open(filePath, 'r') as inputFile:
      return np.load(inputFile)

def plotConfusionMatrix(confusionMatrix, labels, targetPath):
   maxValue = np.max(confusionMatrix, axis=1)

   ax = plt.gca()
   scaled = np.log(((confusionMatrix.T / maxValue).T) + 0.006)

   for(j,i),label in np.ndenumerate(confusionMatrix):
      ax.text(i,j, int(label), ha='center', va='center', color='black',  fontsize=6)

   ticks = np.arange(len(labels))

   ax.set_yticks(ticks)
   ax.set_yticklabels(labels, fontsize=6)
   ax.set_xticks(ticks)
   ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')

   ax.tick_params(axis='both', which='major', bottom=False, top=False, left=False, right=False)

   cmap = plt.get_cmap('Blues')
   cmap_adjusted = colors.LinearSegmentedColormap.from_list('trunc(' + cmap.name +', ' + str(0) + ',' + str(1) + ')', cmap(np.linspace(0,0.8,100)))
   plt.imshow(scaled, cmap=cmap_adjusted, interpolation='none')
   plt.savefig(targetPath, bbox_inches='tight')

   plt.close()

def plotKMeansOverview(data, targetPath, plotDots):
   data = data[data[:,0].argsort()]

   k = data[:,0]

   uniqueK = np.unique(k)

   logger.debug(uniqueK)
   counter = 0
   summed = []
   while counter < uniqueK.shape[0]:
      currentK = uniqueK[counter]
      currentSum = np.sum(data[data[:,0] == currentK], axis=0)
      currentSum = currentSum / data[data[:,0] == currentK].shape[0]

      summed.append(currentSum)
      counter += 1

   summed = np.array(summed)
   meanAveragePrecision = data[:,1]

   correct = data[:,2] / (data[:,2] + data[:,3])

   fig, ax = plt.subplots()

   maximumPrecision = np.argmax(meanAveragePrecision)
   maximumAccuracy = np.argmax(correct)

   if plotDots:
      ax.plot(k, correct, 'go', k, meanAveragePrecision, 'bo')
   ax.plot(summed[:,0], summed[:,2] / (summed[:,2] + summed[:,3]), 'g', label="Accuracy")
   ax.plot(summed[:,0], summed[:,1], 'b', label="Mean average precision")
   ax.plot((k[0], k[::-1][0]),(correct[maximumAccuracy],correct[maximumAccuracy]), 'g--', (k[0], k[::-1][0]),(meanAveragePrecision[maximumPrecision],meanAveragePrecision[maximumPrecision]), 'b--')
   ax.set_xlabel('K')
   ax.axis([k[0], k[::-1][0], 0, 1])
   ax.grid(True)

   plt.legend()
   plt.savefig(targetPath, bbox_inches='tight')
   plt.close()

def splitTestFeaturesByLabel(STest, labelCount):

   activationsByLabel = [[] for i in range(labelCount)]
   counter = 0
   while counter < labelCount:
      currentLabelIndex = STest.shape[1] - labelCount + counter
      currentSelection = STest[test[:, currentLabelIndex] == 1]
      activationsByLabel[counter] = currentSelection
      counter += 1

   return activationsByLabel
