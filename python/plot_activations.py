import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import modules.arachne_caffe as ac

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def getLabelStrings(filePath):
   with open(filePath, 'r') as inputFile:
      result = []
      for line in inputFile.readlines():
        result.append(line.split(' ')[0])

   return result

def plotActivations(activations, labelCount, indexLabelMappingPath, plotFileName):

   firstLabelIndex = activations.shape[1] - labelCount
   activationsPerLabel = 1024 / labelCount
   logger.info(str(activationsPerLabel) + " pixel per label.")

   labels = np.array(activations)[:,firstLabelIndex:]
   selection = np.empty((0, activations.shape[1]))

   tickLabels = []

   labelCounter = 0
   while labelCounter < labels.shape[1]:
      picked = np.random.randint(0,activations.shape[0],2)
      monoLabelSelection = activations[np.logical_or.reduce([activations[:,firstLabelIndex+labelCounter] == 1])]
      picked = np.random.randint(0,monoLabelSelection.shape[0],activationsPerLabel)
      subSelection = monoLabelSelection[picked]
      selection = np.vstack((selection, subSelection))

      tickLabels.append("Label " + str(labelCounter))
      labelCounter += 1

   if indexLabelMappingPath != None:
       tickLabels = getLabelStrings(indexLabelMappingPath)

   scaled = selection[:,0:firstLabelIndex].reshape(selection.shape[0],4,1024)
   scaled = scaled.mean(axis=1)
   maxValue = np.amax(scaled)
   scaled = scaled * (255 / maxValue)

   plt.imshow(scaled, 'afmhot', interpolation='none')
   ax = plt.gca()
   ticks = np.arange(activationsPerLabel * 0.5,selection.shape[0],activationsPerLabel)

   ax.set_yticks(ticks)
   ax.set_yticklabels(tickLabels)

   plt.savefig(plotFileName)
   plt.show()

if __name__ == '__main__':

   indexLabelMappingPath = None

   activationsPath = ""
   if len(sys.argv) != 3 and len(sys.argv) != 4:
      logger.info("Please provide as argument:")
      logger.info("1) npy-file with activations.")
      logger.info("2) The the number of neurons.")
      logger.info("3) The path to the index label mapping (optional).")
      sys.exit()

   activationsPath = sys.argv[1]
   neurons = int(sys.argv[2])

   if len(sys.argv) == 4:
       indexLabelMappingPath = sys.argv[3]

   activations = ac.activationsFromFile(activationsPath)

   labelCount = activations[0,neurons:].shape[0]
   plotActivations(activations, labelCount, indexLabelMappingPath, os.path.splitext(activationsPath)[0] + '_plot.pdf')
