import sys
import os

trainPath = 'label_index_info_train.txt'
testPath = 'label_index_info_test.txt'
mappingPath = 'label_index_mapping.txt'

if(len(sys.argv) < 3):
   print ('Provide the path to a elastic search image dump as argv[1] and a log file as argv[2]')
   sys.exit()

dumpRootPath = sys.argv[1]
logPath = sys.argv[2]

labelCardinality = 0 # Label cardinality is the average number of labels per example in the set
labelDensity = 0 # label density is the number of labels per sample divided by the total number of labels, averaged over the samples

def getCardinality(infoPath, labelCount):
   labelSum = 0
   density = 0
   with open(infoPath) as input:
      counter = 0
      for line in input.readlines():
         split = line.split()
         labelSum += len(split[1:])
         counter += 1
      labelCardinality = float(labelSum) / counter
      density = float(float(labelSum) / labelCount) / counter
      return [labelCardinality, density]

labelCount = 0
with open(dumpRootPath + mappingPath) as input:
   labelCount = len(input.readlines())

cardinalityTraining, densityTraining = getCardinality(dumpRootPath + trainPath, labelCount)
cardinalityTest, densityTest = getCardinality(dumpRootPath + testPath, labelCount)

with open(logPath, 'a') as out:
   print 'writing results to file ' + os.path.abspath(logPath)
   out.write('stats for ' + dumpRootPath + ' with ' + str(labelCount) + ' labels\n')
   out.write('training\tlabel cardinality: ' + str(cardinalityTraining) + ', label density: ' + str(densityTraining) + '\n')
   out.write('test\t\tlabel cardinality: ' + str(cardinalityTest) + ', label density: ' + str(densityTest) + '\n')
   out.write('\n')
