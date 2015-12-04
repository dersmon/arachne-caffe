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
      httpRequest = URL.urlopen(host + url)
      if asJson:
         return json.loads(httpRequest.read())
      else:
         return httpRequest.read()
   except URL.HTTPError as e:
      print("HTTPError for " + host + url)
      print(e)
      return None

def retreiveEntityIds(queries):
   print "Retreiving entity IDs."
   entityIds = []

   for query in queries:
      response = sendQuery("/search?" + query['query'] + "&limit=10", True)
      for entity in response['entities']:
         entityIds.append([entity['entityId'], query['labels']])

   return entityIds

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
               labels.append(image[1])
         dictionary[image[0]] = labels
      else:
         dictionary[image[0]] = image[1]

   return dictionary

def streamFiles(exportFolder, dictionary):

   nthAsTestImage = 5
   counter = 0
   lastPercent = -1

   deadlinkLog = exportFolder + "/invalid_imageIds.txt"
   trainPath = exportFolder + "/train/"
   testPath = exportFolder + "/test/"

   if not os.path.exists(os.path.dirname(trainPath)):
      os.makedirs(os.path.dirname(trainPath))

   if not os.path.exists(os.path.dirname(testPath)):
      os.makedirs(os.path.dirname(testPath))

   print ("\nDownloading images, every "+ str(nthAsTestImage) + "th is beeing picked as a test image.")

   for imageId, labels in dictionary.items():
      image = sendQuery("/image/" + str(imageId), False)

      if image == None:
         with open(deadlinkLog, "a") as log:
            log.write(str(imageId)  + "\n")
         continue

      imageFileName =  str(imageId) + ".jpg"
      targetPath = ""
      infoPath = ""

      if counter % nthAsTestImage == 0:
         targetPath = testPath + imageFileName
         infoPath = exportFolder + "/label_index_info_test.txt"
      else:
         targetPath = trainPath + imageFileName
         infoPath = exportFolder + "/label_index_info_train.txt"

      labelInfoString = targetPath
      for label in labels:
         labelInfoString += " " + label
      labelInfoString += "\n"

      with open(targetPath, "w+") as out:
         out.write(image)

      with open(infoPath, "a") as info:
         info.write(labelInfoString)

      counter += 1
      percent = int((float(counter) / float(len(dictionary))) * 100)

      if percent - lastPercent > 0:
         lastPercent = percent
         print (str(lastPercent) + "%\tdone.")

   print ("100 %\tdone.")

host = "http://arachne.dainst.org/data"
configPath = sys.argv[1]
configJSON = []

with open(configPath, "r") as configuration:
   configJSON = json.load(configuration)

entityIds = retreiveEntityIds(configJSON["queries"])
imageIds = retreiveImageIds(entityIds)
imageDictionary = createImageDictionary(imageIds)

streamFiles("dumps/" + configJSON["exportName"], dictionary)
#retreiveData(urlList)
