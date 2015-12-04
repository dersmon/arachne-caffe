import sys
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
         if image['imageId'] == 0:
            print entity

   return imageIds

def createImageDictionary(imageIds):
   print "Filtering image IDs for dictionary."
   dictionary = dict()

   for image in imageIds:
      if image[0] in dictionary:
         labels = dictionary[image[0]]
         for label in image[1]:
            if label in labels:
               print "Found duplicate: " + str(image)
               continue
            else:
               labels.append(image[1])
         dictionary[image[0]] = labels
      else:
         dictionary[image[0]] = image[1]

   return dictionary

host = "http://arachne.dainst.org/data"
configPath = sys.argv[1]
configJSON = []

with open(configPath, "r") as configuration:
   configJSON = json.load(configuration)

entityIds = retreiveEntityIds(configJSON["queries"])
imageIds = retreiveImageIds(entityIds)
dictionary = createImageDictionary(imageIds)

print len(entityIds)
print len(imageIds)
print len(dictionary)
print dictionary
#retreiveData(urlList)
