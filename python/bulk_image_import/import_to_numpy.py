import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.arachne_caffe as ac
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

trainingDataInfoPath = 'label_index_info_train.txt'
testDataInfoPath = 'label_index_info_test.txt'
labelIndexMappingPath = 'label_index_mapping.txt'

trainingActivationsPath = './training.npy'
testActivationsPath = './test.npy'

batchSize = 200 # how many images to feed to to caffe as one batch
batchLimit = 0  # optional, how many batches should be processed (0 = until no more images)

def calculateActivationVectors(trainingDataInfoPath, testDataInfoPath, labelIndexMappingPath, trainingActivationsPath, testActivationsPath):

      labelCount = 0
      with open(labelIndexMappingPath, 'r') as labelMappingFile:
         for line in labelMappingFile:
            labelCount += 1

      ac.activationsToFile(ac.crunchDumpFiles(trainingDataInfoPath, batchSize, batchLimit, labelCount),trainingActivationsPath)
      ac.activationsToFile(ac.crunchDumpFiles(testDataInfoPath, batchSize, batchLimit, labelCount), testActivationsPath)

if __name__ == '__main__':
   if(len(sys.argv) != 4):
      logger.info("Please provide as arguments:")
      logger.info("1) path to dump root directory as argv[1]")
      logger.info("2) filename training numpy data as argv[2]")
      logger.info("3) filename test numpy data as argv[3]")
      sys.exit
   else:
      trainingDataInfoPath = sys.argv[1] + trainingDataInfoPath
      testDataInfoPath = sys.argv[1] + testDataInfoPath
      labelIndexMappingPath = sys.argv[1] + labelIndexMappingPath
      trainingActivationsPath = "./" + sys.argv[2]
      testActivationsPath = "./" + sys.argv[3]

   calculateActivationVectors(trainingDataInfoPath, testDataInfoPath, labelIndexMappingPath, trainingActivationsPath, testActivationsPath)
