import _mysql
import sys
import getpass

databaseBaseURL = ""
configuration = [["bauwerk",
["SELECT `arachneentityidentification`.`ArachneEntityID`, `marbilder`.`Pfad` " 
+ "FROM `arachneentityidentification`, `marbilder` WHERE `arachneentityidentification`.`ForeignKey` = `marbilder`.`FS_BauwerkID` ORDER BY rand() LIMIT 5000;", "SELECT `arachneentityidentification`.`ArachneEntityID`, `marbilder`.`Pfad`  FROM `arachneentityidentification`, `marbilder` WHERE `arachneentityidentification`.`ForeignKey` = `marbilder`.`FS_BauwerkID` ORDER BY rand() LIMIT 5000;"]
],["topographie", 
["SELECT `arachneentityidentification`.`ArachneEntityID`, `marbilder`.`Pfad`  FROM `arachneentityidentification`, `marbilder` WHERE `arachneentityidentification`.`ForeignKey` = `marbilder`.`FS_TopographieID` ORDER BY rand() LIMIT 5000;"]]]


def startBulkImport():
	fetchData()
	streamFiles()
	
def fetchData():	
	user =     raw_input("Please type in user name for " +databaseBaseURL+ ": ")
	password = getpass.getpass("Password: ")	
	database = raw_input("Database: ")
	
	for target in configuration :		
		print target[0]
		try:
			con = _mysql.connect(databaseBaseURL, user, password, database)
				
			#con.query("SELECT VERSION()")
			#result = con.use_result()
			
			#print "MySQL version: %s" % \
				#result.fetch_row()[0]
			
		except _mysql.Error, e:
		  
			print "Error %d: %s" % (e.args[0], e.args[1])
			sys.exit(1)

		finally:
			
			if con:
				con.close()

		
		print "Todo"

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


	
