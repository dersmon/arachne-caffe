import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.arachne_caffe as ac

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def getLabels(rootPath):
   labels = []
   for directory, _, files in os.walk(rootPath):
      if os.path.basename(directory) == "":
         continue

      labels.append(os.path.basename(directory))
   return labels

def getImages(rootPath, label):
   imageFolder = rootPath + "/" + label

   images = {'label':label, 'files':[]}
   for directory, _, files in os.walk(imageFolder):
      images['files'] = files

   return images

def writeInfoFile(rootPath, images):
   nthTest = 5
   trainInfoPath = rootPath + '/label_index_info_train.txt'
   testInfoPath = rootPath + '/label_index_info_test.txt'
   if not os.path.exists(os.path.dirname(trainInfoPath)):
      os.makedirs(os.path.dirname(trainInfoPath))

   with open(trainInfoPath, 'a') as trainOutput:
      with open(testInfoPath, 'a') as testOutput:
         categoryCounter = 0
         for category in images:
            imageCounter = 0
            for image in category['files']:

               imagePath = rootPath + "/" + category['label'] + "/" + image
               if imageCounter % 5 == 0:
                  testOutput.write(imagePath + ' ' + str(categoryCounter) + '\n')
               else:
                  trainOutput.write(imagePath + ' ' + str(categoryCounter) + '\n')

               imageCounter += 1

            categoryCounter += 1

   return [trainInfoPath, testInfoPath]


if __name__ == '__main__':

   rootPath = ''

   if(len(sys.argv) != 2):
      logger.info("Please provide as argument:")
      logger.info("1) path to root folder")
      sys.exit
   else:
      rootPath = sys.argv[1]

   labels = getLabels(rootPath)

   indexLabelMappingPath = rootPath + '/label_index_mapping.txt'
   if not os.path.exists(os.path.dirname(indexLabelMappingPath)):
      os.makedirs(os.path.dirname(indexLabelMappingPath))

   with open(indexLabelMappingPath, 'a') as output:
      for index, value in enumerate(labels):
         output.write(value + ' ' + str(index) + '\n')

   images = []
   for label in labels:
      images.append(getImages(rootPath, label))

   [trainingInfo, testInfo] = writeInfoFile(rootPath, images)

   ac.activationsToFile(ac.crunchDumpFiles(trainingInfo, 100, 0, len(labels)), './numpy_vectors/handwritten_125_train.npy')
   ac.activationsToFile(ac.crunchDumpFiles(testInfo, 100, 0, len(labels)), './numpy_vectors/handwritten__125_test.npy')
