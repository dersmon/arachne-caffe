import sys
import os
import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def plotComparison(data, targetPath):

   accuracyData = []
   meanAveragePrecisionData = []
   labels = []

   for item in data:
      accuracyData.append(item['accuracy'])
      meanAveragePrecisionData.append(item['meanAveragePrecision'])
      labels.append(item['label'].split('_')[-1])

   logger.debug(accuracyData)
   logger.debug(meanAveragePrecisionData)
   logger.debug(labels)

   fig, ax = plt.subplots()

   index = np.arange(len(data))
   bar_width = 0.25
   opacity = 0.4


   rects1 = plt.barh(index + (2 * bar_width), accuracyData, bar_width,
                 color='g')

   rects2 = plt.barh(index + bar_width, meanAveragePrecisionData, bar_width,
                 color='b')

   labelAccuracy = mpatches.Patch(color='g', label='Accuracy')
   labelMeanAveragePrecision = mpatches.Patch(color='b', label='Mean average precision')

   plt.legend(handles=[labelAccuracy, labelMeanAveragePrecision], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
   ax.tick_params(axis='both', which='major', bottom=True, top=True, left=False, right=False)
   ax.tick_params(axis='both', which='major', bottom=True, top=True, left=False, right=False)
   plt.xlabel('Scores')
   plt.yticks(index + (2 * bar_width), labels)
   plt.xticks(np.arange(0,1.1, 0.1))
   plt.legend()

   ax.xaxis.grid(True)

   plt.savefig(targetPath + "overview.pdf", bbox_inches='tight')

   plt.show()

if __name__ == '__main__':
   if len(sys.argv) != 2:
      logger.info("Please provide as argument:")
      logger.info("1) Path to overview csv files.")
      sys.exit();

   csvFiles = []
   for rootPath, subdirs, files in os.walk(sys.argv[1]):
      for f in files:
         if f.endswith('.csv'):
            csvFiles.append(rootPath + f)

   data = []
   for path in csvFiles:
      with open(path, "r") as inputFile:
         result = np.loadtxt(inputFile, delimiter=",")

         if len(result.shape) == 1:
            data.append({'label':os.path.basename(path).split('.')[0], 'meanAveragePrecision': np.max(result[1]), 'accuracy':np.max(result[2] / (result[2] + result[3]))})
         else:
            data.append({'label':os.path.basename(path).split('.')[0], 'meanAveragePrecision': np.max(result[:,1]), 'accuracy':np.max(result[:,2] / (result[:,2] + result[:,3]))})

   logger.debug(data)

   sortedData = sorted(data, key=lambda k: k['label'])[::-1]

   plotComparison(sortedData, sys.argv[1])
