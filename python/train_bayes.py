import sys
import os
import numpy as np

import modules.utility as utility
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def trainBayes(training, labelCount):

   neurons = training.shape[1] - labelCount
   activationsByLabel = []
   counter = 0
   while counter < labelCount:
      currentLabelIndex = training.shape[1] - labelCount + counter
      currentSelection = training[training[:, currentLabelIndex] == 1][:,0:neurons]
      currentSelection = np.sum(currentSelection, axis=0)
      activationsByLabel.append(currentSelection)
      counter += 1

   activationsByLabel = np.array(activationsByLabel)

   devisor = np.tile(np.sum(activationsByLabel, axis=1),(neurons, 1)).T

   P = (activationsByLabel + 1) / (devisor + neurons)

   PLog = np.log(P)

   # logger.debug(PLog)
   return PLog
