# -*- coding: UTF-8 -*-
import sys
import os
import urllib2 as URL
import json
'''
1) Definiere elastic-search Anfragen und zugehoerige Kategorien.
   1.1) Definition in externer Datei?
   1.2) HTTP Basic Authentifizierung?
2) Sammle sich daraus ergebende Bilder.
3) Erstelle "Woerterbuch" aus Bildern, falls Bild schon vorhanden: mehrere Kategorien.
'''

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

def retreiveEntityIds(queries):
   global limitEntityQuery
   print "Retreiving entity IDs."
   entityIds = []
   labelMapping = []
   for query in queries:
      response = sendQuery("/search?" + query['query'] + "&limit=" + str(limitEntityQuery), True)
      for entity in response['entities']:
         entityIds.append([entity['entityId'], query['labels']])
      for label in query['labels']:
         if label not in labelMapping:
            labelMapping.append(label)
   return [entityIds,labelMapping]

def retreiveImageIds(entityIds):
   print "Retreiving image IDs."
   imageIds = []

   for entity in entityIds:
      response = sendQuery("/entity/" + str(entity[0]), True)
      for image in response['images']:
         imageIds.append([image['imageId'], entity[1]])

   return imageIds

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

   invalidLogPath = exportFolder + "/invalid_imageIds.txt"
   trainingFolderPath = exportFolder + "/train/"
   testFolderPath = exportFolder + "/test/"

   if not os.path.exists(os.path.dirname(trainingFolderPath)):
      os.makedirs(os.path.dirname(trainingFolderPath))

   if not os.path.exists(os.path.dirname(testFolderPath)):
      os.makedirs(os.path.dirname(testFolderPath))

   print ("\nDownloading images, every "+ str(nthAsTestImage) + "th is beeing picked as a test image.")

   trainingInfoPath = exportFolder + "/label_index_info_train.txt"
   testInfoPath = exportFolder + "/label_index_info_test.txt"

   for imageId, labels in dictionary.items():

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
host = "http://arachne.dainst.org/data"
limitEntityQuery = 1000
# end

configPath = sys.argv[1]
configJSON = []

with open(configPath, "r") as configuration:
   configJSON = json.load(configuration)

targetPath = "dumps/" + configJSON["exportName"]

[entityIds, labelMapping] = retreiveEntityIds(configJSON["queries"])

imageIds = retreiveImageIds(entityIds)

imageDictionary = createImageDictionary(imageIds)
indexLabelMappingPath = targetPath + "/label_index_mapping.txt"

if not os.path.exists(os.path.dirname(indexLabelMappingPath)):
   os.makedirs(os.path.dirname(indexLabelMappingPath))

counter = 0
for label in labelMapping:
   with open(indexLabelMappingPath, "a") as output:
      output.write(label + " " + str(counter) + "\n")
      counter += 1

streamFiles(targetPath, imageDictionary, labelMapping)

print "Entities: " + str(len(entityIds)) + ", images:" + str(len(imageIds)) + ", image dictionary: " + str(len(imageDictionary))
