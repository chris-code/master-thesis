import numpy as np

def sigmoid(x):
	#return x
	return 1.0 / (1.0 + np.exp(-x))

def generate_w(size):
	return np.concatenate([np.full(size//2, 1.0), np.full(size//2, -1.0)])

def generate_i(size):
	i_orig = np.full(size, i_value)
	i_orig[np.arange(size) % 2 == 1] = -i_value
	i_orig[-asymmetric_portion:] = i_value
	return i_orig

coeff = 0.001
asymmetric_portion = 4
i_value = 1.0
dimensions = [2**x for x in range(5, 21+1)]
results = []

for dim in dimensions:
	w = generate_w(dim)
	i_orig = generate_i(dim)
	i_ae = i_orig + coeff * np.sign(w)
	
	i_diff = np.mean(np.abs(i_orig - i_ae))

	inner_prod_orig = np.dot(w, i_orig)
	inner_prod_ae = np.dot(w, i_ae)
	value_orig = sigmoid(inner_prod_orig)
	value_ae = sigmoid(inner_prod_ae)

	results.append( (dim, i_diff, inner_prod_orig, inner_prod_ae, value_orig, value_ae) )
	print('Dim= {0}\tOriginal: {4:.4f}\tAdversarial: {5:.4f}'.format(*results[-1]))

with open('ae_explanation_data.dat', 'w') as out_file:
	out_file.write('# dim, i_diff, inner_prod_orig, inner_prod_ae, value_orig, value_ae\n')
	for r in results:
		out_file.write('{0} {1} {2} {3} {4} {5}\n'.format(*r))
