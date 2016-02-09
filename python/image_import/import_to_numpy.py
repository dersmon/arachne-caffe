import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.arachne_caffe as ac
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

batchSize = 200 # how many images to feed to to caffe as one batch
batchLimit = 0  # optional, how many batches should be processed (0 = until there are no more images)

def calculateActivationVectors(trainingDataInfoPath, testDataInfoPath, labelIndexMappingPath, trainingActivationsPath, testActivationsPath):

      labelCount = 0
      with open(labelIndexMappingPath, 'r') as labelMappingFile:
         for line in labelMappingFile:
            labelCount += 1

      ac.activationsToFile(ac.crunchDumpFiles(trainingDataInfoPath, batchSize, batchLimit, labelCount),trainingActivationsPath)
      ac.activationsToFile(ac.crunchDumpFiles(testDataInfoPath, batchSize, batchLimit, labelCount), testActivationsPath)

if __name__ == '__main__':
   if(len(sys.argv) != 6):
      logger.info("Please provide as arguments:")
      logger.info("1) path to label_info_training.txt as argv[1]")
      logger.info("2) path to label_info_test.txt as argv[2]")
      logger.info("3) path to label_index_mapping.txt as argv[3]")
      logger.info("4) target-filename for training data (*.npy) as argv[4]")
      logger.info("5) target-filename for test data (*.npy) as argv[5]")
      sys.exit
   else:
      trainingDataInfoPath = sys.argv[1]
      testDataInfoPath = sys.argv[2]
      labelIndexMappingPath = sys.argv[3]
      trainingActivationsPath = sys.argv[4]
      testActivationsPath = sys.argv[5]

   calculateActivationVectors(trainingDataInfoPath, testDataInfoPath, labelIndexMappingPath, trainingActivationsPath, testActivationsPath)
