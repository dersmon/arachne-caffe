import sys
import logging
import os
import json
import elastic_query
import import_statistics
import random

limitEntityQuery = 1000

imagePerEntity = 1000
labelMapping = []
targetPath = './image_imports/'
harvestingTest = False
labelDistributionAdjusted = []
adjustmentFactor = 1

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def startBulkDownload(targetPath, configJSON, showStatisticsPlot):

   global labelDistributionAdjusted
   global labelMapping

   logger.info('Target path for image dump: ' + os.path.abspath(targetPath) + '.')

   labelMapping = []
   [entityIds, labelMapping] = retreiveEntityIds(configJSON['queries'], False)

   labelDistributionAdjusted = [1] * len(labelMapping)
   logger.debug(labelDistributionAdjusted)
   logger.info('Found ' + str(len(entityIds)) + ' entities with ' + str(len(labelMapping)) + ' labels.')
   imageIds = retreiveImageIds(entityIds)

   imageDictionary = createImageDictionary(imageIds)

   labelHistogram = [0] * len(labelMapping)
   imagesByLabel = [[] for i in range(len(labelMapping))]

   labelSum = 0
   for key, value in imageDictionary.items():
      labelHistogram[labelMapping.index(value)] += 1
      imagesByLabel[labelMapping.index(value)].append(key)
      labelSum += 1

   logger.info("Initial label distribution:")
   logger.info(labelHistogram)

   imagesPerLabel = min(labelHistogram)

   imageDictionary = pickFromSurplus(imagesByLabel, imagesPerLabel)

   logger.info("Label distribution:")
   logger.info(labelHistogram)
   logger.info(labelMapping)
   logger.info(labelDistributionAdjusted)

   logger.info('Original images: ' + str(len(imageIds)) + ', filtered: ' + str(len(imageDictionary)) + '.')

   logger.info('Writing label index mapping...')

   indexLabelMappingPath = targetPath + '/label_index_mapping.txt'
   if not os.path.exists(os.path.dirname(indexLabelMappingPath)):
      os.makedirs(os.path.dirname(indexLabelMappingPath))

   with open(indexLabelMappingPath, 'a') as output:
      for index, value in enumerate(labelMapping):
         output.write(value + ' ' + str(index) + '\n')


   streamFiles(targetPath, imageDictionary, labelMapping)

   import_statistics.evaluate( targetPath + '/', targetPath + '/' + 'labelinfo.log', targetPath + '/distribution_' + '.pdf' ,  showStatisticsPlot)

def retreiveEntityIds(queries, limitByDistribution):
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

      limit = None
      if limitByDistribution:
         limit = int(limitEntityQuery * labelDistributionAdjusted[labelMapping.index(label)])
      else:
         limit = limitEntityQuery

      response = elastic_query.sendQuery('/search?' + query['query'] + '&limit=' + str(limit), True)
      for entity in response['entities']:
         logger.debug('Received entity ' + str(entity['entityId']) + ', label: ' + label + '.')
         entityIds.append([entity['entityId'], label])

   return [entityIds,labelMapping]

def retreiveImageIds(entityIds):

   logger.info('Retreiving image IDs linked to entities...')
   imageIds = []
   global labelDistributionAdjusted

   for entity in entityIds:
      response = elastic_query.sendQuery('/entity/' + str(entity[0]), True)
      counter = 0
      for image in response['images']:
         if imagePerEntity != 0 and counter < imagePerEntity:
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
         continue
      else:
         dictionary[image[0]] = image[1]

   return dictionary

def pickFromSurplus(imagesByLabel, maxImages):
   logger.info('Picking ' + str(maxImages) + ' images per label.')
   resultDictionary = dict()
   labelCounter = 0
   for images in imagesByLabel:
      random.shuffle(images)
      subList = images[0:maxImages]
      for image in subList:
         resultDictionary[image] = labelMapping[labelCounter]

      labelCounter += 1
   return resultDictionary

def streamFiles(exportFolder, dictionary, labelMapping):

   logger.info('Downloading image files...')

   nthAsTestImage = 5
   counter = 0
   lastPercent = -1

   invalidLogPath = exportFolder + '/invalid_image_ids.txt'
   trainingFolderPath = exportFolder + '/train/'
   testFolderPath = exportFolder + '/test/'
   
   for label in labelMapping:

      if not os.path.exists(os.path.dirname(trainingFolderPath + label + '/')):
         os.makedirs(os.path.dirname(trainingFolderPath + label + '/'))
         logger.debug('Created folder: ' + trainingFolderPath + label + '/')

      if not os.path.exists(os.path.dirname(testFolderPath + label + '/')):
         os.makedirs(os.path.dirname(testFolderPath + label + '/'))
         logger.debug('Created folder: ' + testFolderPath + label + '/')

   if  harvestingTest == False:
      logger.info('Downloading images, every '+ str(nthAsTestImage) + 'th is beeing picked as a test image.')
   else:
      logger.info('Skipping image downloads. Just writing index info files.')

   trainingInfoPath = exportFolder + '/label_info_training.txt'
   testInfoPath = exportFolder + '/label_info_test.txt'

   for imageId, label in dictionary.items():

      if harvestingTest== False:
         imageData = elastic_query.sendQuery('/image/' + str(imageId), False)

         if imageData == None:
            with open(invalidLogPath, 'a') as log:
               log.write(str(imageId)  + '\n')
            continue

      imageFileName =  str(imageId) + '.jpg'
      targetPath = ''
      infoPath = None

      if counter % nthAsTestImage == 0:
         targetPath = testFolderPath + label + '/' + imageFileName
         infoPath = testInfoPath
      else:
         targetPath = trainingFolderPath + label + '/' + imageFileName
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

   logger.info('\nDone.')

if __name__ == '__main__':
   logger.info('Running harvesting test (just collecting metadata, skipping image download): ' + str(harvestingTest))
   configPath = sys.argv[1]
   configJSON = []

   with open(configPath, 'r') as configuration:
      configJSON = json.load(configuration)

   targetPath += configJSON['exportName'] + '_' + str(limitEntityQuery)
   startBulkDownload(targetPath, configJSON, True)
