import logging
import urllib2
import base64
import json

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

host = 'http://bogusman01.dai-cloud.uni-koeln.de/data'

def sendQuery(url, asJson):
   logging.debug('Elastic search host: ' + host)
   logging.debug('Query: ' + url + ', returning as json: ' + str(asJson))
   try:
      url = urllib2.quote(url.encode('utf8'), '/:"&=?')

      auth_handler = urllib2.HTTPBasicAuthHandler()
      auth_handler.add_password(None, uri=(host + url), user='user', passwd='passwd')
      opener = urllib2.build_opener(auth_handler)

      urllib2.install_opener(opener)
      result = urllib2.urlopen(host + url)

      logger.debug(result)

      if asJson:
         return json.loads(result.read())
      else:
         return result.read()

   except urllib2.HTTPError as e:
      logger.debug(e)
      logger.debug(host + url)
      return None
