import logging
import numpy as np

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def getLabelStrings(filePath):
   with open(filePath, 'r') as inputFile:
      result = []
      for line in inputFile.readlines():
        result.append(line.split(' ')[0])

   return result

def arrayFromFile(filePath):
	logger.info('Opening file: ' + filePath)
	with open(filePath, 'r') as inputFile:
		return np.load(inputFile)
