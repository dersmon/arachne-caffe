import _mysql
import sys
import getpass
import xml.etree.ElementTree as ET
import urllib2 as URL
import os

configurationFilePath = ""
databaseBaseURL = "crazyhorse.archaeologie.uni-koeln.de"
databaseName = "arachne"

exportsRoot = "./exports/"

nthForTesting = 5

exportFolder = ""
configuration = []breakfast scene
data = []

def startBulkImport():
	loadConfig()
	fetchData()
	streamFiles()
	
def loadConfig():
	global exportFolder
	global configurationFilePath
	global configuration
	
	try:
		e = ET.parse(configurationFilePath).getroot()
		
		exportFolder = e.attrib['exportName']
		
		for category in e:
			currentCategory = category.attrib['label']
			queryStrings = []
			for query in category:
				queryStrings.append(query.text)
			
			configuration.append([currentCategory, queryStrings])
			
		print("Configuration loaded.")
	
	except IOError as e:
		print("Could not load export configuration at " + configurationFilePath)
		sys.exit()
	
def fetchData():		
	global exportFolder
	global configuration
	
	user =     raw_input("Please type in user name for " + databaseBaseURL + ": ")
	password = getpass.getpass("Password: ")	
	
	con = _mysql.connect(databaseBaseURL, user, password, databaseName)
				
	labelIndex = 0
			
	mappingPath = exportsRoot + exportFolder + "/indexLabelMapping.txt"
	
	if not os.path.exists(os.path.dirname(mappingPath)):
		os.makedirs(os.path.dirname(mappingPath))
		
	for target in configuration :	
		print ("Querying database for label: " + target[0])
		
		counter = 0	
		
		for mySQLString in target[1]:
			try:
				con.query(mySQLString)
				result = con.store_result()
				
				row = result.fetch_row()
				
				while(row):
					data.append(ImageInfo(row[0][0], row[0][1], target[0], labelIndex))
					counter += 1
					row = result.fetch_row()
								
			except _mysql.Error as e:			  
				print ("Error %d: %s" % (e.args[0], e.args[1]))
				sys.exit(1)
		
		print("Retreived " + str(counter) + " image paths.")
		
		with open(mappingPath, "a") as mapping:
			mapping.write(str(labelIndex) + ": " + target[0] + "\n")	
			
		labelIndex += 1

	if con:
		con.close()


def streamFiles():
	global exportFolder
	global configuration
	
	count = 0
	lastPercent = -1	
	
	deadlinkLog = exportsRoot + exportFolder + "/deadLinks.txt"	
	trainPath = exportsRoot + exportFolder + "/train/"
	testPath = exportsRoot + exportFolder + "/test/"
	
	if not os.path.exists(os.path.dirname(trainPath)):
		os.makedirs(os.path.dirname(trainPath))
		
	if not os.path.exists(os.path.dirname(testPath)):
		os.makedirs(os.path.dirname(testPath))
	
	print ("\nDownloading images, every "+ str(nthForTesting) + "th is beeing picked as a test image.")
	
	for imageInfo in data:
		
		try:
			image = URL.urlopen(imageInfo.sourcePath)
			imageFileName = imageInfo.sourcePath.split('/')[-1]
			targetPath = ""
			infoPath = ""
			
			if count % nthForTesting == 0:
				targetPath = testPath + imageFileName
				infoPath = testPath + "labelIndexInfo.txt"
			else:
				targetPath = trainPath + imageFileName				
				infoPath = trainPath + "labelIndexInfo.txt"
	
			
			with open(targetPath, "w+") as out:
				out.write(image.read())
			
			with open(infoPath, "a") as info:
				info.write(targetPath + " " + str(imageInfo.labelIndex) + "\n")	
				
		except URL.HTTPError as e:			  
			print("URL Error for " + imageInfo.sourcePath + ", server returned 404.")
			with open(deadlinkLog, "a") as log:
				log.write(imageInfo.sourcePath  + "\n")	
			
		except URL.URLError as e: 			
			print("URL Error for " + imageInfo.sourcePath + ", no answer.")
		
		count += 1		
		percent = int((float(count) / float(len(data))) * 100)
		
		if percent - lastPercent > 0:
			lastPercent = percent
			print (str(lastPercent) + "% done.")
	
	print ("Done.")
	
class ImageInfo:
	'Container class for image information.'
	
	def __init__(self, arachneEntityID, sourcePath, label, labelIndex):
		self.arachneEntityID = arachneEntityID
		self.sourcePath = sourcePath
		self.label = label
		self.labelIndex = labelIndex
	
	def printContents(self):
		print (self.targetPath, ": Arachne Entity ID: ", self.arachneEntityID, ", ", self.sourcePath)
	
if(len(sys.argv) != 2):
	print("Please provide path to bulk export config as argument.")
	sys.exit(1)

configurationFilePath = sys.argv[1]


startBulkImport()
