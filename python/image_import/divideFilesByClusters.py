import sys
import os
import numpy as np
import logging
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.utility as utility
import clustering.kMeans_mixed as kMeans

from shutil import copyfile


logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

K = 5

if __name__ == '__main__':
   if(len(sys.argv) != 4 and len(sys.argv) != 5):
      logger.info("Please provide as arguments:")
      logger.info("1) activations (*.npy).")
      logger.info("2) info file (*.txt)")
      logger.info("3) target path for sorted images")
      logger.info("4) cluster pickle (optional)")
      sys.exit()

   targetPath = sys.argv[3]
   if targetPath.endswith('/') == False:
      targetPath += '/'

   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))

   imagePaths = []

   with open(sys.argv[2], "r") as result:
      for line in result.readlines():

         split = line.strip().split(' ')
         imagePaths.append(split[0])
   # for rootPath, subdirs, files in os.walk(sys.argv[2]):
   #    for f in files:
   #       if f.endswith('.jpg'):
   #          imagePaths.append(rootPath + f)

   clusters = None
   if len(sys.argv) == 5:
      clusters = pickle.load(sys.argv[4])

   activations = utility.arrayFromFile(sys.argv[1])

   if clusters == None:
      [clusters, iterations] = kMeans.findKMeans(activations, K, 0, targetPath)

   for clusterIndex, cluster in enumerate(clusters):
      for index in cluster['memberIndices']:
         currentTarget = targetPath + str(clusterIndex) + "/"

         if not os.path.exists(os.path.dirname(currentTarget)):
            os.makedirs(os.path.dirname(currentTarget))

         copyfile(imagePaths[index], currentTarget + os.path.basename(imagePaths[index]))
