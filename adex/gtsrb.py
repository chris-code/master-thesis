import subprocess
import numpy as np
import caffe

def load_model(layout_path, weight_path, batch_size):
    net = caffe.Net(layout_path,
                    weight_path,
                    caffe.TEST)
    shape = list(net.blobs['data'].data.shape)
    shape[0] = batch_size
    net.blobs['data'].reshape(*shape)
    net.blobs['prob'].reshape(batch_size, )
    net.reshape()
    
    return net

def build_transformer(net):
    transformer = caffe.io.Transformer( {'data': net.blobs['data'].data.shape} )
    transformer.set_transpose('data', (2,0,1))
    #transformer.set_raw_scale('data', 255) # the reference model operates on images in [0,255] range instead of [0,1]
    #transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB
    
    return transformer

def load_image(transformer, path):
    image = caffe.io.load_image(path)
    image = transformer.preprocess('data', image)
    image = np.expand_dims(image, 0)
    
    return image
