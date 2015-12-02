import httplib
import base64
import string
import json
import codecs

'''
1) Definiere elastic-search Anfragen und zugehoerige Kategorien.
   1.1) Definition in externer Datei?
   1.2) HTTP Basic Authentifizierung?
2) Sammle sich daraus ergebende Bilder.
3) Erstelle "Woerterbuch" aus Bildern, falls Bild schon vorhanden: mehrere Kategorien.
'''

def sendQuery(url):

   global host

   webservice = httplib.HTTP(host)
   webservice.putrequest("GET", url)

   webservice.putheader("Host", host)
   webservice.putheader("User-Agent", "Python httplib")
   webservice.putheader("Accept", "application/json;")
   webservice.putheader("Content-Type", "application/json;")
   webservice.endheaders()

   statuscode, statusmessage, header = webservice.getreply()
   res = webservice.getfile().read()
   # print "Content: ", res
   # print "Response: ", statuscode, statusmessage
   # print "Headers: ", header

   return [statuscode, statusmessage, res]

def retreiveData(queryURLs):
   for url in queryURLs:
      [statusCode, statusMessage, response] = sendQuery(url)
      response = json.loads(response)

      with open('data.json', 'w') as outfile:
          json.dump(response, outfile)

host = "arachne.dainst.org"
urlList = ["/data/search?fq=facet_kategorie:\"Einzelobjekte\"&q=*"]

retreiveData(urlList)
