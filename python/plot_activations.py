import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import modules.utility as utility
import matplotlib.colors as colors


logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def plotOverview(activations, labelCount, indexLabelMappingPath, plotFileName):
   firstLabelIndex = activations.shape[1] - labelCount
   reduceActivationsBy = 1
   if firstLabelIndex % 4 == 0 and firstLabelIndex > 1024:
      reduceActivationsBy = 4
   elif firstLabelIndex % 2 == 0 and firstLabelIndex > 1024:
      reduceActivationsBy = 2

   reduceActivationsTo = firstLabelIndex / reduceActivationsBy

   activationsPerLabel = 1024 / labelCount

   logger.info(str(activationsPerLabel) + " rows per label.")

   labels = np.array(activations)[:,firstLabelIndex:]
   selection = np.empty((0, activations.shape[1]))

   tickLabels = []

   labelCounter = 0
   while labelCounter < labels.shape[1]:
      picked = np.random.randint(0,activations.shape[0],2)
      monoLabelSelection = activations[np.logical_or.reduce([activations[:,firstLabelIndex+labelCounter] == 1])]
      picked = np.random.randint(0, monoLabelSelection.shape[0], activationsPerLabel)
      subSelection = monoLabelSelection[picked]
      selection = np.vstack((selection, subSelection))

      tickLabels.append("Label " + str(labelCounter))
      labelCounter += 1

   if indexLabelMappingPath != None:
       tickLabels = utility.getLabelStrings(indexLabelMappingPath)

   cmap = plt.get_cmap('Greys_r')
   cmap_adjusted = colors.LinearSegmentedColormap.from_list('trunc(' + cmap.name +', ' + str(0) + ',' + str(1) + ')', cmap(np.linspace(0,1,100)))

   scaled = np.reshape(selection[:,0:firstLabelIndex], (selection.shape[0], reduceActivationsBy, reduceActivationsTo))
   scaled = scaled.mean(axis=1)

   plt.imshow(scaled, cmap=cmap_adjusted, interpolation='none')

   ax = plt.gca()
   # ax.pcolormesh(scaled, cmap=plt.get_cmap('afmhot'))
   ax.tick_params(axis='both', which='major', bottom=False, top=False, left=False, right=False)
   ticks = np.arange(activationsPerLabel * 0.5,selection.shape[0],activationsPerLabel)

   ax.set_yticks(ticks)
   ax.set_yticklabels(tickLabels)

   plt.savefig(plotFileName, bbox_inches='tight')

def plotLabelSummaries():
   logger.info('Todo...')

def plotActivations(activations, labelCount, targetFolder, indexLabelMappingPath):
   plotOverview(activations, labelCount, indexLabelMappingPath, targetFolder + '_overview.pdf')
   plotLabelSummaries()

if __name__ == '__main__':

   indexLabelMappingPath = None

   activationsPath = ""
   if len(sys.argv) != 4 and len(sys.argv) != 5:
      logger.info("Please provide as argument:")
      logger.info("1) npy-file with activations.")
      logger.info("2) The the number of neurons.")
      logger.info("3) The target path.")
      logger.info("3) The path to the index label mapping (optional).")
      sys.exit()

   activationsPath = sys.argv[1]
   neurons = int(sys.argv[2])
   targetPath = sys.argv[3]

   if len(sys.argv) == 5:
       indexLabelMappingPath = sys.argv[4]

   activations = utility.arrayFromFile(activationsPath)

   if targetPath.endswith('/') == False:
      targetPath += '/'

   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   labelCount = activations[0,neurons:].shape[0]
   plotActivations(activations, labelCount, targetPath, indexLabelMappingPath)
