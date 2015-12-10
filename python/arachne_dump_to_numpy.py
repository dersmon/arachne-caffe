import modules.arachne_caffe as ac

trainingDataInfoPath = './dumps/elastic_test_small/label_index_info_train.txt'
testDataInfoPath = './dumps/elastic_test_small/label_index_info_test.txt'
labelIndexMappingPath = './dumps/elastic_test_small/label_index_mapping.txt'

trainingActivationsPath = './training_vectors_elastic_small.npy'
testActivationsPath = './test_vectors_elastic_small.npy'

labelCount = 0
with open(labelIndexMappingPath, 'r') as labelMappingFile:
   for line in labelMappingFile:
      labelCount += 1

batchSize = 200 # how many images to feed to to caffe as one batch
batchLimit = 0  # optional, how many batches should be processed (0 = until no more images)

ac.activationsToFile(ac.crunchDumpFiles(trainingDataInfoPath, batchSize, batchLimit, labelCount), trainingActivationsPath)
ac.activationsToFile(ac.crunchDumpFiles(testDataInfoPath, batchSize, batchLimit, labelCount), testActivationsPath)
