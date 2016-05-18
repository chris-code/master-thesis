import numpy as np

# Get the network's predictions and the corresponding probabilities.
# Parameter data: Shape is (batch_size, 3, xres, yres)
# Output predictions: Shape is (batch_size, #classes), sorted by likelihood
# Output probabilities: Shape is (batch_size, #classes)
def predict(net, data):
    net.blobs['data'].data[...] = data
    probabilities = net.forward()['prob']
    predictions = probabilities.argsort()[:,::-1]
    
    return predictions, probabilities

# Compute the gradient of the loss function w.r.t. the data, while using desired_labels as target labels.
# Computing the gradient with caffe requires a forward pass to be explicitly performed. If you already did
# a forward pass, instruct this function to skip it by passing do_forward_pass=False
# Parameter data: Shape is (batch_size, 3, xres, yres)
# Parameter desired_labels: Shape is (batch_size)
# Output gradient: Same shape as input data
def compute_gradient(net, data, desired_labels, do_forward_pass=True):
    if do_forward_pass:
        _, _ = predict(net, data)
    probs = np.zeros_like(net.blobs['prob'].data)
    probs[np.arange(probs.shape[0]), desired_labels] = 1

    gradient = net.backward(prob=probs, diffs=['data'])
    
    return gradient['data'].copy() #TODO .copy() required?

def make_adversarial(net, data, desired_labels, ae_grad_coeff, confidence_target, max_iterations):
    adversarial_data = data.copy()
    
    for i in range(max_iterations):
        predictions, probabilities = predict(net, adversarial_data)
        confidence = np.mean(probabilities[np.arange(len(desired_labels)), desired_labels])
        if confidence >= confidence_target:
            break
        
        gradient = compute_gradient(net, adversarial_data, desired_labels, do_forward_pass=False)
        adversarial_data = adversarial_data + (ae_grad_coeff / max_iterations) * np.sign(gradient)
    
    return adversarial_data, confidence, i
