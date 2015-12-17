import sys
import modules.arachne_caffe as ac


trainingDataInfoPath = 'label_index_info_train.txt'
testDataInfoPath = 'label_index_info_test.txt'
labelIndexMappingPath = 'label_index_mapping.txt'

trainingActivationsPath = './training.npy'
testActivationsPath = './test.npy'


batchSize = 200 # how many images to feed to to caffe as one batch
batchLimit = 0  # optional, how many batches should be processed (0 = until no more images)

if(len(sys.argv) != 4):
   print("Please provide as arguments:")
   print("1) path to dump root directory")
   print("2) filename training numpy data")
   print("3) filename test numpy data")
else:
   trainingDataInfoPath = sys.argv[1] + trainingDataInfoPath
   testDataInfoPath = sys.argv[1] + testDataInfoPath
   labelIndexMappingPath = sys.argv[1] + labelIndexMappingPath
   trainingActivationsPath = "./" + sys.argv[2]
   testActivationsPath = "./" + sys.argv[3]

labelCount = 0
with open(labelIndexMappingPath, 'r') as labelMappingFile:
   for line in labelMappingFile:
      labelCount += 1

ac.activationsToFile(ac.crunchDumpFiles(trainingDataInfoPath, batchSize, batchLimit, labelCount), trainingActivationsPath)
ac.activationsToFile(ac.crunchDumpFiles(testDataInfoPath, batchSize, batchLimit, labelCount), testActivationsPath)
