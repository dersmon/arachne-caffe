import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import modules.arachne_caffe as ac

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def plotActivations(activations, labelCount):

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

   scaled = selection[:,0:firstLabelIndex].reshape(selection.shape[0],4,1024)
   scaled = scaled.mean(axis=1)
   maxValue = np.amax(scaled)
   scaled = scaled * (255 / maxValue)

   plt.imshow(scaled, 'Greys_r', interpolation='none')
   ticks = np.arange(activationsPerLabel * 0.5,selection.shape[0],activationsPerLabel)

   ax.set_yticks(ticks)
   ax.set_yticklabels(tickLabels)

   plt.show()

if __name__ == '__main__':

   activationsPath = ""
   if len(sys.argv) != 3:
      logger.info("Please provide as argument:")
      logger.info("1) npy-file with activations as.")
      logger.info("2) The the number of neurons.")
      sys.exit()

   activationsPath = sys.argv[1]
   neurons = int(sys.argv[2])
   activations = ac.activationsFromFile(activationsPath)

   labelCount = activations[0,neurons:].shape[0]
   plotActivations(activations, labelCount)
