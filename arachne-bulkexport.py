import _mysql
import sys
import getpass
import xml.etree.ElementTree as ET

configurationFilePath = "./arachne-bulkexport-config.xml"
databaseBaseURL = "crazyhorse.archaeologie.uni-koeln.de"
databaseName = "arachne_hohl"

configuration = []
data = []

def startBulkImport():
	loadConfig()
	fetchData()
	streamFiles()
	
def loadConfig():
	e = ET.parse(configurationFilePath).getroot()
	for category in e:
		currentCategory = category.attrib['label']
		queryStrings = []
		for query in category:
			queryStrings.append(query.text)
		
		configuration.append([currentCategory, queryStrings])
	
	print "Configuration loaded."
	
def fetchData():	
	user =     raw_input("Please type in user name for " +databaseBaseURL+ ": ")
	password = getpass.getpass("Password: ")	
	
	con = _mysql.connect(databaseBaseURL, user, password, databaseName)
				
	for target in configuration :	
		print "Querying database for label: " + target[0]
		
		counter = 0	
		
		for mySQLString in target[1]:
			try:
				con.query(mySQLString)
				result = con.store_result()
				
				row = result.fetch_row()
				
				while(row):
					data.append(ImageInfo(row[0][0], row[0][1], target[0]))
					counter += 1
					row = result.fetch_row()
								
			except _mysql.Error, e:
			  
				print "Error %d: %s" % (e.args[0], e.args[1])
				sys.exit(1)
		
		print "Found " + str(counter) + " images."

	if con:
		con.close()


def streamFiles():
	print "Todo"

	
class ImageInfo:
	'Container class for image data.'
	arachneEntityID = ""
	sourcePath = ""
	targetPath = ""
	
	def __init__(self, arachneEntityID, sourcePath, targetPath):
		self.arachneEntityID = arachneEntityID
		self.sourcePath = sourcePath
		self.targetPath = targetPath
	
	def printContents(self):
		print self.targetPath, " - Arachne Entity ID: ", self.arachneEntityID, ", ", self.sourcePath
	

startBulkImport()


	
