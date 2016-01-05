import sys
import logging
import os
import urllib2 as URL
import json
import elastic_query
import download_statistics

limitEntityQuery = 100
limitImagePerEntity = 3
labelMapping = []
targetPath = './image_dumps/'
harvestingTest = True

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def retreiveEntityIds(queries):
   logger.info('Retreiving entity IDs...')
   global limitEntityQuery
   global labelMapping

   entityIds = []

   for query in queries:
      logger.debug('Query:')
      logger.debug(query)
      label = query['label']
      if label not in labelMapping:
         labelMapping.append(label)

      response = elastic_query.sendQuery('/search?' + query['query'] + '&limit=' + str(limitEntityQuery), True)
      for entity in response['entities']:
         logger.debug('Received entity ' + str(entity['entityId']) + ', label: ' + label + '.')
         entityIds.append([entity['entityId'], label])

   return [entityIds,labelMapping]

def retreiveImageIds(entityIds):
   logger.info('Retreiving image IDs linked to entities...')
   imageIds = []

   for entity in entityIds:
      response = elastic_query.sendQuery('/entity/' + str(entity[0]), True)
      counter = 0
      for image in response['images']:
         if limitImagePerEntity != 0 and counter < limitImagePerEntity:
            logger.debug('Received image ' + str(image['imageId']) + ', label: ' + str(entity[1]) + '.')
            imageIds.append([image['imageId'], entity[1]])
            counter += 1
         else:
            break

   return imageIds

def createImageDictionary(imageIds):
   logger.info('Removing duplicate images...')
   dictionary = dict()
   for image in imageIds:
      if image[0] in dictionary:
         labels = dictionary[image[0]]
         for label in image[1]:
            if label in labels:
               logger.debug('Found duplicate: ' + str(image))
               continue
            else:
               labels.append(label)
         dictionary[image[0]] = labels
      else:
         dictionary[image[0]] = image[1]

   return dictionary

def streamFiles(exportFolder, dictionary, labelMapping):

   nthAsTestImage = 5
   counter = 0
   lastPercent = -1

   invalidLogPath = exportFolder + '/invalid_image_ids.txt'
   trainingFolderPath = exportFolder + '/train/'
   testFolderPath = exportFolder + '/test/'

   logger.debug('Path to training images: ' + trainingFolderPath)
   if not os.path.exists(os.path.dirname(trainingFolderPath)):
      os.makedirs(os.path.dirname(trainingFolderPath))

   logger.debug('Path to test images: ' + testFolderPath)
   if not os.path.exists(os.path.dirname(testFolderPath)):
      os.makedirs(os.path.dirname(testFolderPath))

   if  harvestingTest == False:
      logger.info('\nDownloading images, every '+ str(nthAsTestImage) + 'th is beeing picked as a test image.')
   else:
      logger.info('\nSkipping image downloads. Just writing index info files.')

   trainingInfoPath = exportFolder + '/label_index_info_train.txt'
   testInfoPath = exportFolder + '/label_index_info_test.txt'

   for imageId, label in dictionary.items():

      if harvestingTest == False:
         imageData = elastic_query.sendQuery('/image/' + str(imageId), False)

         if imageData == None:
            with open(invalidLogPath, 'a') as log:
               log.write(str(imageId)  + '\n')
            continue

      imageFileName =  str(imageId) + '.jpg'
      targetPath = ''
      infoPath = None

      if counter % nthAsTestImage == 0:
         targetPath = testFolderPath + imageFileName
         infoPath = testInfoPath
      else:
         targetPath = trainingFolderPath + imageFileName
         infoPath = trainingInfoPath

      labelInfoString = targetPath + ' ' + str(labelMapping.index(label)) + '\n'

      if harvestingTest == False:
         with open(targetPath, 'w+') as out:
            out.write(imageData)

      with open(infoPath, 'a') as info:
         info.write(labelInfoString)

      counter += 1
      percent = int((float(counter) / float(len(dictionary))) * 100)

      if percent - lastPercent > 0:
         lastPercent = percent
         logger.info(str(lastPercent) + '%\tdone.')

   logger.info('100%\tdone.')

if __name__ == '__main__':

   logger.info('Running harvesting test (just collecting metadata, skipping image download): ' + str(harvestingTest))
   configPath = sys.argv[1]
   configJSON = []

   with open(configPath, 'r') as configuration:
      configJSON = json.load(configuration)

   targetPath += configJSON['exportName'] + '_' + str(limitEntityQuery)

   logger.info('Target path for image dump: ' + os.path.abspath(targetPath) + '.')

   [entityIds, labelMapping] = retreiveEntityIds(configJSON['queries'])

   logger.info('Found ' + str(len(entityIds)) + ' entities with ' + str(len(labelMapping)) + ' labels.')
   imageIds = retreiveImageIds(entityIds)

   imageDictionary = createImageDictionary(imageIds)
   logger.info('Original images: ' + str(len(imageIds)) + ', filtered: ' + str(len(imageDictionary)) + '.')
   logger.info('Writing label index mapping...')

   indexLabelMappingPath = targetPath + '/label_index_mapping.txt'
   if not os.path.exists(os.path.dirname(indexLabelMappingPath)):
      os.makedirs(os.path.dirname(indexLabelMappingPath))

   with open(indexLabelMappingPath, 'a') as output:
      for index, value in enumerate(labelMapping):
         output.write(value + ' ' + str(index) + '\n')

   logger.info('Downloading image files...')
   streamFiles(targetPath, imageDictionary, labelMapping)

   download_statistics.evaluate(targetPath + '/', targetPath + '/' + 'labelinfo.log', True)
