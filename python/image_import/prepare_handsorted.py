import sys
import os
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def subFolders(rootPath):
   labels = []
   for directory, _, files in os.walk(rootPath):
      if os.path.basename(directory) == "":
         continue

      labels.append(os.path.basename(directory))

   labels.sort()
   return labels

def getImages(rootPath, folder, hierarchy):
   imageFolder = rootPath + "/" + folder
   label = ''
   if hierarchy == 0:
      label = folder
   elif hierarchy == 1:
      label = folder.split('_')[0]

   images = {'label':label, 'folder':folder, 'files':[]}

   for directory, _, files in os.walk(imageFolder):
      images['files'] = files

   return images

def writeInfoFile(rootPath, images, uniqueLabels):
   nthTest = 5
   trainInfoPath = rootPath + 'label_index_info_train_' + str(hierarchy) +'.txt'
   testInfoPath = rootPath + 'label_index_info_test_' + str(hierarchy) +'.txt'
   if not os.path.exists(os.path.dirname(trainInfoPath)):
      os.makedirs(os.path.dirname(trainInfoPath))

   labels = []

   for image in images:
      labels.append(image['label'])

   # logger.debug(labels)


   with open(trainInfoPath, 'a') as trainOutput:
      with open(testInfoPath, 'a') as testOutput:
         categoryCounter = 0
         for category in images:
            imageCounter = 0
            for image in category['files']:

               imagePath = rootPath + category['folder'] + "/" + image
               if imageCounter % 5 == 0:
                  testOutput.write(imagePath + ' ' + str(uniqueLabels.index(labels[categoryCounter])) + '\n')
               else:
                  trainOutput.write(imagePath + ' ' + str(uniqueLabels.index(labels[categoryCounter])) + '\n')

               imageCounter += 1

            categoryCounter += 1

   return [trainInfoPath, testInfoPath]


if __name__ == '__main__':

   rootPath = ''
   hierarchy = 0

   if(len(sys.argv) != 2):
      logger.info("Please provide as argument:")
      logger.info("1) path to root folder")
      sys.exit
   else:
      rootPath = sys.argv[1]

   subFolders = subFolders(rootPath)

   images = []
   for folder in subFolders:
      images.append(getImages(rootPath , folder, hierarchy))

   uniqueLabels = []
   for image in images:
      uniqueLabels.append(image['label'])

   uniqueLabels = list(set(uniqueLabels))
   uniqueLabels.sort()
   logger.debug(uniqueLabels)

   indexLabelMappingPath = rootPath + '/label_index_mapping_' + str(hierarchy) +'.txt'
   if not os.path.exists(os.path.dirname(indexLabelMappingPath)):
      os.makedirs(os.path.dirname(indexLabelMappingPath))

   with open(indexLabelMappingPath, 'a') as output:
      for index, value in enumerate(uniqueLabels):
         output.write(value + ' ' + str(index) + '\n')


   [trainingInfo, testInfo] = writeInfoFile(rootPath, images, uniqueLabels)
   exportName = os.path.dirname(rootPath)
   exportName = exportName.split('/')[len(exportName.split('/')) - 1]
