import subprocess
import numpy as np
import caffe

def load_model(caffe_root, batch_size):
    net = caffe.Net(caffe_root + '/models/bvlc_googlenet/deploy.prototxt',
                    caffe_root + '/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                    caffe.TEST)
    shape = list(net.blobs['data'].data.shape)
    shape[0] = batch_size
    net.blobs['data'].reshape(*shape)
    net.blobs['prob'].reshape(batch_size, )
    net.reshape()
    
    return net


def load_labels(caffe_root):
    location = '/data/ilsvrc12/synset_words.txt'
    
    try:
        labels = np.loadtxt(caffe_root + location, str, delimiter='\t')
    except IOError: # If the data is not found, let this caffe script download it
        subprocess.call(caffe_root + '/data/ilsvrc12/get_ilsvrc_aux.sh')
        labels = np.loadtxt(caffe_root + location, str, delimiter='\t')

    new_labels = []
    for l in labels:
        new_l = l[10:].split(',')
        new_l.insert(0, l[:10])
        new_labels.append(new_l)
    
    return new_labels

def get_label_from_class_name(labels, classname):
    classname = classname.strip()
    
    for index, l in enumerate(labels):
        if l[0].strip() == classname:
            return index
    else:
        raise KeyError('Class name not found')

def build_transformer(net):
    transformer = caffe.io.Transformer( {'data': net.blobs['data'].data.shape} )
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255) # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB
    
    return transformer

def load_image(transformer, path):
    image = caffe.io.load_image(path)
    image = transformer.preprocess('data', image)
    image = np.expand_dims(image, 0)
    
    return image
