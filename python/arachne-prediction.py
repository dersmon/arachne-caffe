import sys
import caffe
import cv2
import Image
import numpy as np
from scipy.misc import imresize

root = './'
caffe_root = '/home/simon/Workspaces/caffe/'
#MODEL_FILE = caffe_root + 'models/placesCNN/places205CNN_deploy.prototxt'
#PRETRAINED = caffe_root + 'models/placesCNN/places205CNN_iter_300000.caffemodel'

MODEL_FILE = root + 'trained_models/custom_alex/deploy.prototxt'
PRETRAINED = root + 'trained_models/custom_alex/caffe_alexnet_train_iter_1500.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
mean_file = np.array([104,117,123]) 
transformer.set_mean('data', mean_file) #### subtract mean ####
transformer.set_raw_scale('data', 255) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR


blob = caffe.proto.caffe_pb2.BlobProto()
data = open(root + 'trained_models/custom_alex/mean_train.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)

meanArray = np.array( caffe.io.blobproto_to_array(blob) ).transpose(3,2,1,0)
meanArray = meanArray[:,:,:,0]

# Test self-made image

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(root + 'example_images/Statue_of_Liberty_Paris_001.jpg'))


# out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
out = net.forward()
print("Predicted class is #{}.".format(out['prob'][0].argmax()))

# load labels
#imagenet_labels_filename = caffe_root + 'models/placesCNN/categoryIndex_places205.csv'
imagenet_labels_filename = root + 'trained_models/custom_alex/indexLabelMapping.txt'

labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\s')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1: -6: -1]
print out['prob'][0]
print top_k
print labels[top_k]
