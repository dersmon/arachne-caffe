import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import modules.utility as utility

import clustering.kMeans_core as kMeans_core
import clustering.kMeans_mixed as kMeans_mixed
import clustering.kMeans_per_label as kMeans_per_label

import plot_cluster as plot_cluster

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS_PER_LABEL = 150
MAX_ITERATIONS_MIXED = 150

PER_LABEL_START = 1
PER_LABEL_END = 8

MIXED_START = 32
MIXED_STEP = 8
MIXED_END = 128

RUNS_PER_TYPE = 3

def generateKMeansSeries(activationsPath, labelIndexMappingPath, targetFolder):
   if targetFolder.endswith('/') == False:
      targetFolder += '/'

   activations = utility.arrayFromFile(activationsPath)
   labels = utility.getLabelStrings(labelIndexMappingPath)
   neurons = activations.shape[1] - len(labels)

   # test per label k-means
   k = PER_LABEL_START
   while k <= PER_LABEL_END:
      runCounter = 0
      while runCounter < RUNS_PER_TYPE:

         logger.info("Calculating per label KMeans with k = " + str(k) + ".")
         currentTarget = targetFolder + "perLabel_" + str(k) + "/run_" + str(runCounter) + "/"
         if not os.path.exists(os.path.dirname(currentTarget)):
            os.makedirs(os.path.dirname(currentTarget))

         c, i = kMeans_per_label.findKMeansPerLabel(activations, k, len(labels), currentTarget, labelIndexMappingPath)
         plot_cluster.plotClusters(kMeans_core.cleanUp(c), i, neurons, currentTarget, labels[:len(labels)])
         runCounter += 1
      k += 1


   # test mixed k-means
   k = MIXED_START
   while k <= MIXED_END:
      runCounter = 0
      while runCounter < RUNS_PER_TYPE:

         logger.info("Calculating mixed KMeans with k = " + str(k) + ".")
         currentTarget = targetFolder + "mixed_" + str(k) + "/run_" + str(runCounter) + "/"
         if not os.path.exists(os.path.dirname(currentTarget)):
            os.makedirs(os.path.dirname(currentTarget))

         [c,i] = kMeans_mixed.findKMeans(activations, k, len(labels), currentTarget)
         plot_cluster.plotClusters(kMeans_core.cleanUp(c), i, activations.shape[1] - len(labels), currentTarget, labels[:len(labels)])
         runCounter += 1
      k += MIXED_STEP

if __name__ == '__main__':
   if len(sys.argv) != 4:

      logger.info("Please provide as argument:")
      logger.info("1) Path to training activations (*.npy).")
      logger.info("2) Path to label_index_mapping (*.txt).")
      logger.info("3) Target path for evaluation results.")

      sys.exit();

   generateKMeansSeries(sys.argv[1], sys.argv[2], sys.argv[3])
