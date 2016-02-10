import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import operator
import random
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

trainPath = 'label_info_training.txt'
testPath = 'label_info_test.txt'
mappingPath = 'label_index_mapping.txt'

def getCardinalityAndDensity(infoPath, labelCount):
   labelSum = 0
   density = 0
   histogram = np.array([0] * labelCount)
   with open(infoPath) as input:
      counter = 0
      for line in input.readlines():
         split = line.split()
         splitLabels = [int(x) for x in split[1:]]
         histogram[splitLabels] += 1
         labelSum += len(split[1:])
         counter += 1
      cardinality = float(labelSum) / counter
      density = cardinality / labelCount
      return [cardinality, density, histogram]

def evaluate(dumpRootPath, logPath, plotPath, showPlot):

   # see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
   labelCardinality = 0
   labelDensity = 0



   labelCount = 0
   labelMapping = []
   with open(dumpRootPath + mappingPath) as input:
      for line in input.readlines():
         temp = line.split()[0:len(line.split()) - 1]
         temp2 = ""

         for fragmet in temp:
            temp2 += fragmet + " "

         labelMapping.append(temp2.strip().lower().decode('utf8'))
         labelCount += 1

   cardinalityTraining, densityTraining, histogramTraining = getCardinalityAndDensity(dumpRootPath + trainPath, labelCount)
   cardinalityTest, densityTest, histogramTest = getCardinalityAndDensity(dumpRootPath + testPath, labelCount)

   labelDictionary = dict()
   for index, value in enumerate(labelMapping):
      labelDictionary[value.encode('utf8')] = histogramTraining[index]

   labelDictionarySorted = sorted(labelDictionary.items(), key=operator.itemgetter(1))

   logger.debug(labelDictionarySorted)

   with open(logPath, 'a') as out:
      out.write(os.path.abspath(dumpRootPath) + '\n')
      out.write('labels\t\t' + str(labelCount) + '\n')
      out.write('training\tlabel cardinality: ' + str(cardinalityTraining) + ', label density: ' + str(densityTraining) + '\n')
      out.write('training labels:\n')
      for index, value in enumerate(labelDictionarySorted):
         out.write(value[0] + ': '+ str(value[1]) + '\n')
      out.write('test\t\tlabelprint cardinality: ' + str(cardinalityTest) + ', label density: ' + str(densityTest) + '\n')
      out.write('test labels:\n')
      for index, value in enumerate(labelMapping):
         out.write(value.encode('utf8') + ': '+ str(histogramTest[index]) + '\n')
      out.write('\n')

      logger.info('training: label cardinality: ' + str(cardinalityTraining) + ', label density: ' + str(densityTraining))
      logger.info('test\t: label cardinality: ' + str(cardinalityTest) + ', label density: ' + str(densityTest))
      logger.info('Writing complete label statistics to file ' + os.path.abspath(logPath))

   logger.info('Saving label distribution plot to file ' + os.path.abspath(plotPath))
   y_pos = np.arange(len(labelMapping))
   plt.barh(y_pos, histogramTraining, align='center')
   plt.yticks(y_pos, labelMapping)
   plt.title('Label distribution:')
   plt.savefig(plotPath)

   if showPlot:
      plt.show()

def createLabelHistogram(imagesByLabel, labels, evaluationTargetPath):

   histogramLabel = np.array([0] * len(labels))

   for labelIndex, images in enumerate(imagesByLabel):
      histogramLabel[labelIndex] = len(images)

   y_pos = np.arange(len(labels))
   x_pos = np.arange(0, np.max(histogramLabel), 50)

   plt.barh(y_pos, histogramLabel, height=0.8, left=None, hold=None, align='center')
   plt.yticks(y_pos, labels)
   plt.xticks(x_pos, x_pos)
   plt.gca().xaxis.grid(True)
   # plt.title('Label distribution:')
   # plt.savefig(evaluationTargetPath + "label_distribution.pdf", bbox_inches='tight')
   plt.show()
   plt.close()

def createExampleGrids(imagesByLabel, labels, evaluationTargetPath):

   for labelIndex, images in enumerate(imagesByLabel):
      random.shuffle(images)
      fig = plt.figure(figsize = (9, 9))
      spec = gridspec.GridSpec(3, 3)
      spec.update(wspace=0.01, hspace=0.01)
      imageCount = 0
      while (imageCount < 9):
         imageSub = fig.add_subplot(spec[imageCount])
         img = mpimg.imread(images[imageCount])

         imgPlot = plt.imshow(img, cmap=cm.Greys_r)

         imageSub.axes.get_xaxis().set_visible(False)
         imageSub.axes.get_yaxis().set_visible(False)
         imageCount += 1

      fig.suptitle(labels[labelIndex] + ': ' + str(len(images)))

      fig.savefig(evaluationTargetPath + "examples_" + labels[labelIndex] + ".pdf")
      plt.close()



def evaluateImport(infoTrainPath, infoTestPath, labelIndexMappingPath, evaluationTargetPath):
   if evaluationTargetPath.endswith('/') == False:
      evaluationTargetPath += '/'

   if not os.path.exists(os.path.dirname(evaluationTargetPath)):
      os.makedirs(os.path.dirname(evaluationTargetPath))

   # TODO: Redundant code in evaluate_neuralnet
   labels = []
   with open(labelIndexMappingPath, "r") as inputFile:
      for line in inputFile.readlines():
         labels.append(line.split(' ')[0])

   imagesByLabel = [[] for i in range(len(labels))]

   with open(infoTrainPath, "r") as inputFile:
      for line in inputFile.readlines():
         split = line.split(' ')
         imagePath = split[0]
         labelIndex = int(split[1])
         imagesByLabel[labelIndex].append(imagePath)

   with open(infoTestPath, "r") as inputFile:
      for line in inputFile.readlines():
         split = line.split(' ')
         imagePath = split[0]
         labelIndex = int(split[1])
         imagesByLabel[labelIndex].append(imagePath)

   createLabelHistogram(imagesByLabel, labels, evaluationTargetPath)
   # createExampleGrids(imagesByLabel, labels, evaluationTargetPath)

if __name__ == '__main__':
   if(len(sys.argv) != 5):
      logger.error('Required:')
      logger.error('1) Path to training info (*.txt)')
      logger.error('2) Path to test info (*.txt)')
      logger.error('3) Path to label index mapping (*.txt)')
      logger.error('4) Target path for evaluation results.')
      logger.error('Exiting...')
      sys.exit()
   else:
      evaluateImport(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
