import _mysql
import sys
import getpass
import xml.etree.ElementTree as ET

configurationFilePath = "./arachne-bulkexport-config.xml"

databaseBaseURL = ""
configuration = []

def startBulkImport():
	loadConfig()
	fetchData()
	# streamFiles()
	
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
	database = raw_input("Database: ")
	
	for target in configuration :	
		print "Querying database for label: " + target[0]	
		
		for mySQLString in target[1]:
			try:
				con = _mysql.connect(databaseBaseURL, user, password, database)
				con.query(mySQLString)
				result = con.use_result()
				
				print result.fetch_row()[0]
				
			except _mysql.Error, e:
			  
				print "Error %d: %s" % (e.args[0], e.args[1])
				sys.exit(1)

			finally:
				
				if con:
					con.close()


def streamFiles():
	print "Todo"

class QueryInfo:
	targetFolder = ""
	sqlQueries = []
	
	def __init__(self, targetFolder, sqlQueries):
		self.targetFolder = targetFolder
		self.sqlQueries = sqlQueries
	
	def printContents(self):
		print "Target folder:", self.targetFolder
		print "Queries:", str(self.sqlQueries)
		
class ImageInfo:
	arachneEntityID = ""
	imageCounter = 0
	sourcePath = ""
	targetPath = ""
	
	def __init__(self, arachneEntityID, imageCounter, sourcePath, targetPath):
		self.arachneEntityID = arachneEntityID
		self.imageCounter = imageCounter
		self.sourcePath = sourcePath
		self.targetPath = targetPath
	
	def printContents(self):
		print self.targetPath, " - Arachne Entity ID: ", self.arachneEntityID, ", ", self.sourcePath
	

	
if(len(sys.argv) == 1):
	print "Please base URL to database as argv[1]"
elif (len(sys.argv) == 2):
	databaseBaseURL = sys.argv[1]

startBulkImport()


	
