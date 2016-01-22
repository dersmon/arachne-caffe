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

   logger.debug(firstLabelIndex)

   labels = np.array(activations)[:,firstLabelIndex:]
   selection = np.empty((0, activations.shape[1]))

   labelCounter = 0
   while labelCounter < labels.shape[1]:
      subSelection = activations[np.logical_or.reduce([activations[:,firstLabelIndex+labelCounter] == 1])][0:100,:]
      selection = np.vstack((selection, subSelection))
      labelCounter += 1

   scaled = selection[:,0:firstLabelIndex].reshape(selection.shape[0],4,1024)
   scaled = scaled.mean(axis=1)
   maxValue = np.amax(scaled)
   scaled = scaled * (255 / maxValue)

   plt.imshow(scaled, 'Greys_r', interpolation='none')
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
