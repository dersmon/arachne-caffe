import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import operator

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

if __name__ == '__main__':
   if(len(sys.argv) != 4):
      logger.error('Required:')
      logger.error('1) Path to a elastic dump as argv[1]')
      logger.error('2) Path/name for log file as argv[2]')
      logger.error('3) Path/name for plot of labeldistribution as argv[3]')
      logger.error('Exiting...')
      sys.exit()
   else:
      evaluate(sys.argv[1], sys.argv[2], sys.argv[3], True)
