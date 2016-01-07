import logging
import urllib2 as URL
import json

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
      logger.debug(e)
      logger.debug(host + url)
      return None
