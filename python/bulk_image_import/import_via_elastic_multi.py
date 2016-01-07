# -*- coding: UTF-8 -*-
import sys
import os
import urllib2 as URL
import json

def sendQuery(url, asJson):
   global host
   try:
      url = URL.quote(url.encode('utf8'), '/:"&=?')
      httpRequest = URL.urlopen(host + url)

      if asJson:
         return json.loads(httpRequest.read())
      else:
         return httpRequest.read()
   except URL.HTTPError as e:
      # print("HTTPError for " + host + url)
      # print(e)
      return None

def retreiveEntitiesByQuery(queries):
   global limitEntityQuery
   global knownEntities
   print "Retreiving entity IDs."
   entityIds = []
   labelMapping = []
   for query in queries:
      response = sendQuery("/search?" + query['query'] + "&limit=" + str(limitEntityQuery), True)
      print query['query'] + " yielded " + str(len(response['entities'])) + " entities."
      for entity in response['entities']:
         if entity['entityId'] in knownEntities:
            continue
         else:
            knownEntities.append(entity['entityId'])
            entityIds.append([entity['entityId'], []])
   return entityIds

def retreiveImageIds(entityIds):
   global facetList
   global labelMapping

   logPath = "./elastic.log"
   images = []
   for entity in entityIds:
      response = sendQuery("/entity/" + str(entity[0]), True)
      labels = entity[1]
      for facet in facetList:
         if facet in response:
            for facetValue in response[facet]:
               labels = parseLabel(facetValue, labels)

      if "images" in response:
         for image in response["images"]:
            if image["imageId"] == 0:
               # print image
               continue
            currentImage = [image["imageId"], labels]
            images.append(currentImage)

   return images

def retreiveLinkedEntities(imageIds, keepLabels):
   entityIds = []
   for image in imageIds:
      if image[0] == 0:
         print image
      response = sendQuery("/search?q=connectedEntities:" + str(image[0]), True) #&q=connectedEntities%3A1241415
      # print "/search?q=connectedEntities:" + str(image[0])
      # print response["entities"]
      if response != None and "entities" in response:
         for entity in response["entities"]:
            if entity['entityId'] in knownEntities:
               continue
            else:
               if keepLabels:
                  entityIds.append([entity["entityId"],image[1]])
               else:
                  entityIds.append([entity["entityId"],[]])
               knownEntities.append(entity['entityId'])

   return entityIds

def parseLabel(facetValue, existing):
   global labelMapping

   for possible in labelMapping:
      if possible in facetValue.lower() and possible not in existing:
         existing.append(possible)
      #    print possible + " is in " + facetValue
      # else:
      #    print possible + " is not in " + facetValue

   return existing

def createImageDictionary(imageIds):
   print "Filtering image IDs for dictionary."
   dictionary = dict()
   for image in imageIds:
      if image[0] in dictionary:
         labels = dictionary[image[0]]
         for label in image[1]:
            if label in labels:
               #print "Found duplicate: " + str(image)
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

   invalidLogPath = exportFolder + "/invalid_image_ids.txt"
   trainingFolderPath = exportFolder + "/train/"
   testFolderPath = exportFolder + "/test/"

   if not os.path.exists(os.path.dirname(trainingFolderPath)):
      os.makedirs(os.path.dirname(trainingFolderPath))

   if not os.path.exists(os.path.dirname(testFolderPath)):
      os.makedirs(os.path.dirname(testFolderPath))

   if  harvestingTest == False:
      print ("\nDownloading images, every "+ str(nthAsTestImage) + "th is beeing picked as a test image.")
   else:
      print ("\nSkipping image downloads. Just writing index info files.")

   trainingInfoPath = exportFolder + "/label_index_info_train.txt"
   testInfoPath = exportFolder + "/label_index_info_test.txt"

   for imageId, labels in dictionary.items():
      if harvestingTest == False:
         imageData = sendQuery("/image/" + str(imageId), False)
         if imageData == None:
            with open(invalidLogPath, "a") as log:
               log.write(str(imageId)  + "\n")
            continue

      imageFileName =  str(imageId) + ".jpg"
      targetPath = ""
      infoPath = None

      if counter % nthAsTestImage == 0:
         targetPath = testFolderPath + imageFileName
         infoPath = testInfoPath
      else:
         targetPath = trainingFolderPath + imageFileName
         infoPath = trainingInfoPath

      labelInfoString = targetPath

      for label in labels:
         labelInfoString += " " + str(labelMapping.index(label))
      labelInfoString += "\n"
      if harvestingTest == False:
         with open(targetPath, "w+") as out:
            out.write(imageData)

      with open(infoPath, "a") as info:
         info.write(labelInfoString)

      counter += 1
      percent = int((float(counter) / float(len(dictionary))) * 100)

      if percent - lastPercent > 0:
         lastPercent = percent
         print (str(lastPercent) + "%\tdone.")

   print ("100%\tdone.")

# start global variables
host = "http://bogusman01.dai-cloud.uni-koeln.de/data"
limitEntityQuery = 100
facetList = ["facet_kategorie", "facet_material", "facet_objektgattung", "facet_technik", "facet_technik", "facet_kulturkreis", "facet_subkategorie_objekt", "title"]
labelConfigPath = "./examples/dump_configs/elastic/multi/labels.txt"
labelMapping = []
knownEntities = []

harvestingTest = True
# end

configPath = sys.argv[1]
configJSON = []

with open(configPath, "r") as configuration:
   configJSON = json.load(configuration)

with open(labelConfigPath, "r") as labelConfig:
   for line in labelConfig.readlines():
      if(line.strip().lower().decode('utf8') not in labelMapping):
         labelMapping.append(line.strip().lower().decode('utf8'))

targetPath = "dumps/" + configJSON["exportName"]
entityIds = retreiveEntitiesByQuery(configJSON["queries"])
imageIds = []

print "Running harvesting test (e.g. just collecting metadata, skipping image download): " + str(harvestingTest)

# if len(sys.argv) == 2:
#    with open("./dumps/" + configJSON["exportName"] + "/image_dictionary.tsv", "r") as input:
#       for line in input.readlines()
#          imageIds.append(line.strip())

# search by config -> entity ids ohne label
# 1)  retreive image ids -> labels
# 1.1)entitiy ids = retreive linked via entitiy ids -> entity ids ohne label
# 1.2)entitiy ids += retreive linked via image ids -> entity ids mit label
# 2)  retreive image ids -> labels
#     retreive linked via entitiy ids -> z.t. label (wenn hinzugefügt über 1.2)
#     retreive linked via image ids -> entity ids mit label

iteration = 0
print "iteration " + str(iteration) + ", input entities: " + str(len(entityIds)) + ", images: " + str(len(imageIds)) + ", known entities: " + str(len(knownEntities))
while iteration < 5:
   currentImages = retreiveImageIds(entityIds)
   nextEntityIds = retreiveLinkedEntities(currentImages, True)
   nextEntityIds.extend(retreiveLinkedEntities(entityIds, False))
   entityIds = nextEntityIds
   imageIds += currentImages
   iteration += 1
   print "iteration " + str(iteration) + ", input entities: " + str(len(entityIds)) + ", images: " + str(len(imageIds)) + ", new images: " + str(len(currentImages))  + ", known entities: " + str(len(knownEntities))

imageDictionary = createImageDictionary(imageIds)

if not os.path.exists(os.path.dirname("./dumps/" + configJSON["exportName"] + "/image_dictionary.tsv")):
   os.makedirs(os.path.dirname("./dumps/" + configJSON["exportName"] + "/image_dictionary.tsv"))

with open("./dumps/" + configJSON["exportName"] + "/image_dictionary.tsv", "w") as output:
   json.dump(imageDictionary, output)

indexLabelMappingPath = targetPath + "/label_index_mapping.txt"

if not os.path.exists(os.path.dirname(indexLabelMappingPath)):
   os.makedirs(os.path.dirname(indexLabelMappingPath))

counter = 0
for label in labelMapping:
   with open(indexLabelMappingPath, "a") as output:
      output.write(label.encode('utf-8') + " " + str(counter) + "\n")
      counter += 1

streamFiles(targetPath, imageDictionary, labelMapping)

print "Number of entities: " + str(len(entityIds)) + ", images:" + str(len(imageIds)) + ", image dictionary items: " + str(len(imageDictionary))
