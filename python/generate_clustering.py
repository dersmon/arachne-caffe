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
PER_LABEL_END = 12

MIXED_START = 5
MIXED_STEP = 5
MIXED_END = 120



def generateKMeansSeries(activations, labelIndexMappingPath, targetFolder, subfolderPrefix):
   if targetFolder.endswith('/') == False:
      targetFolder += '/'

   activations = utility.arrayFromFile(trainingActivationsPath)
   labels = utility.getLabelStrings(labelIndexMappingPath)
   neurons = activations.shape[1] - len(labels)

   # test per label k-means
   k = PER_LABEL_START
   while k < PER_LABEL_END:
      logger.info("Testing per label KMeans with k = " + str(k) + ".")
      currentTarget = targetFolder + subfolderPrefix + "_perLabel_" + str(k) + "/"
      if not os.path.exists(os.path.dirname(currentTarget)):
         os.makedirs(os.path.dirname(currentTarget))

      c, i = kMeans_per_label.findKMeansPerLabel(activations, k, len(labels), currentTarget, labelIndexMappingPath)
      plot_cluster.plotClusters(kMeans_core.cleanUp(c), i, neurons, currentTarget, labels[:len(labels)])

      k += 1

   # test mixed k-means
   k = MIXED_START
   while k < MIXED_END:
      logger.info("Testing mixed KMeans with k = " + str(k) + ".")
      currentTarget = targetFolder + subfolderPrefix + "_mixed_" + str(k) + "/"
      if not os.path.exists(os.path.dirname(currentTarget)):
         os.makedirs(os.path.dirname(currentTarget))
      [c,i] = kMeans_mixed.findKMeans(activations, k, len(labels), currentTarget)
      plot_cluster.plotClusters(kMeans_core.cleanUp(c), i, trainingActivations.shape[1] - len(labels), currentTarget, labels[:len(labels)])

      k += MIXED_STEP

if __name__ == '__main__':
   if len(sys.argv) != 5:

      logger.info("Please provide as argument:")
      logger.info("1) Path to training activations (*.npy).")
      logger.info("2) Path to label_index_mapping (*.txt).")
      logger.info("3) Target path for evaluation results.")
      logger.info("4) Prefix for result folders.")
      sys.exit();

   generateKMeansSeries(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
