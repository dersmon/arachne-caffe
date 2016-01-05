import logging
import urllib2 as URL
import json

host = 'http://bogusman01.dai-cloud.uni-koeln.de/data'

def sendQuery(url, asJson):
   logging.debug('Elastic search host: ' + host)
   logging.debug('Query: ' + url + ', returning as json: ' + str(asJson))
   try:
      url = URL.quote(url.encode('utf8'), '/:"&=?')
      httpRequest = URL.urlopen(host + url)
      if asJson:
         return json.loads(httpRequest.read())
      else:
         return httpRequest.read()

   except URL.HTTPError as e:
      logging.error(e)
      logging.error(host + url)
      return None
