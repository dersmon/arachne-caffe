import sys
import os
import numpy as np
import logging
import pickle
import caffe

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.utility as utility


logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 200 # how many images to feed to to caffe as one batch
BATCH_LIMIT = 0  # optional, how many batches should be processed (0 = until there are no more images)

root = './'

MODEL_FILE = root + 'caffe_models/hybrid_cnn_handsorted_lmdb/deploy_fc7.prototxt'
PRETRAINED_FILE = root + 'caffe_models/hybridCNN_iter_700000.caffemodel'
# MEAN_FILE = root + 'image_imports/handsorted_lmdb/mean_train.binaryproto'
LAST_LAYER_NAME = 'fc7'
USE_GPU = False

net = None
transformer = None

def setupCaffe():
   global net, transformer

   net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
   if USE_GPU:
      caffe.set_mode_gpu()
   else:
      caffe.set_mode_cpu()

   # blob = caffe.proto.caffe_pb2.BlobProto()
   # data = open(MEAN_FILE).read()
   # blob.ParseFromString(data)
   # mean = np.array(caffe.io.blobproto_to_array(blob))[0].mean(1).mean(1)

   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   transformer.set_transpose('data', (2,0,1))
   transformer.set_raw_scale('data', 255)
   # transformer.set_mean('data', mean)
   transformer.set_channel_swap('data', (2,1,0))

def crunchDumpFiles(imagePaths):
   imageBatchCount = 0
   imageCount = 0
   if net == None or transformer == None:
      setupCaffe()

   activations = []
   currentBatch = []
   currentBatchSize = 0
   batchCount = 0

   for path in imagePaths:

      currentBatch.append(path)
      currentBatchSize += 1

      if currentBatchSize == BATCH_SIZE:
         activations.extend(evaluateImageBatch(currentBatch))

         currentBatchSize = 0
         currentBatch = []
         batchCount += 1

         # logger.info(str(len(activations)))

         if batchCount == BATCH_LIMIT and BATCH_LIMIT != 0:
            break;

   activations.extend(evaluateImageBatch(currentBatch))
   logger.info('Final number of images processed: ' + str(len(activations)))
   return activations


def evaluateImageBatch(imageBatch):
   global net, transformer
   if net == None or transformer == None:
      setupCaffe()

   batchActivations = []
   imageData = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)) , imageBatch)

   net.blobs['data'].reshape(len(imageBatch), net.blobs['data'].shape[1], net.blobs['data'].shape[2], net.blobs['data'].shape[3])
   net.blobs['data'].data[...] = imageData

   out = net.forward()

   counter = 0
   for image in imageBatch:
      batchActivations.append(out[LAST_LAYER_NAME][counter])
      counter += 1

   return batchActivations

if __name__ == '__main__':
   if(len(sys.argv) != 3):
      logger.info("Please provide as arguments:")
      logger.info("1) path to image folder.")
      logger.info("2) target path for activations file (*.npy)")
      sys.exit()


   imagePaths = []
   targetPath = sys.argv[2]

   for rootPath, subdirs, files in os.walk(sys.argv[1]):
      for f in files:
         if f.endswith('.jpg'):
            imagePaths.append(rootPath + f)

   activations = crunchDumpFiles(imagePaths)
   
   if not os.path.exists(os.path.dirname(targetPath)):
      os.makedirs(os.path.dirname(targetPath))
   with open(targetPath, "w") as output:
      np.save(output, activations)
