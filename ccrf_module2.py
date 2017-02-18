import numpy as np
from scipy import stats
from scipy.sparse import *
from scipy import *
import math

def learning(feature_filename, edge_filename, edge_filename2):

	filepath = feature_filename

	data = np.loadtxt(fname=filepath, delimiter='\t')
	data = stats.zscore(data)

	relation_filepath = edge_filename

	relation_sparse = np.loadtxt(fname=relation_filepath, delimiter='\t')

	row = relation_sparse[:,0]
	col = relation_sparse[:,1]
	weight = relation_sparse[:,2]

	R = csr_matrix((weight,(row,col))).todense()
	R = np.transpose(R)

	relation_filepath = edge_filename2

	relation_sparse = np.loadtxt(fname=relation_filepath, delimiter='\t')

	row = relation_sparse[:,0]
	col = relation_sparse[:,1]
	weight = relation_sparse[:,2]

	R2 = csr_matrix((weight,(row,col))).todense()
	R2 = np.transpose(R2)


	y = np.transpose(np.matrix(data[:,0]))
	X = np.matrix(data[:,1:])

	num_data, num_x_feature = X.shape

	num_y_feature = 2
	num_feature = num_x_feature + num_y_feature

	alpha = np.matrix(1 * np.ones((num_x_feature, 1)))
	alpha_new = np.matrix(alpha)
	delta_log_alpha = np.matrix(0.0 * np.ones((num_x_feature, 1)))

	beta = 0.1
	beta_new = beta

	beta2 = 0.1
	beta2_new = beta2

	n = num_data

	D_r_1 = np.matrix(np.diag(np.squeeze(np.asarray(R.sum(axis=1)))))
	D_c_1 = np.matrix(np.diag(np.squeeze(np.asarray(R.sum(axis=0)))))

	D_r_2 = np.matrix(np.diag(np.squeeze(np.asarray(R2.sum(axis=1)))))
	D_c_2 = np.matrix(np.diag(np.squeeze(np.asarray(R2.sum(axis=0)))))


	iterations = 100
	learning_rate = 0.000001
	precision = 0.0017

	for i in range(iterations):
		a = np.transpose(alpha) * np.matrix(np.ones((alpha.shape[0], 1)))
		b = 2*X*alpha + beta*(D_r_1-D_c_1) * np.matrix(np.ones((D_r_1.shape[0],1))) + beta2*(D_r_2-D_c_2) * np.matrix(np.ones((D_r_2.shape[0],1)))
		b_t = np.transpose(b)

		for k in range(num_x_feature):
			x_k = X[:,k]
			alpha_k = alpha[k]
			log_alpha_k = math.log(alpha_k)

			delta_log_alpha_k = alpha_k*(n/(2*a) + (1/(4*a**2))*b_t*b - (1/(2*a))*b_t*x_k  + np.transpose(x_k)*x_k - np.transpose(y-x_k)*(y-x_k))
			delta_log_alpha[k] = delta_log_alpha_k
			log_alpha_k = log_alpha_k + learning_rate*delta_log_alpha_k
			alpha_new[k] = math.exp(log_alpha_k)
	
		delta_beta = -1/(2*a)*b_t*(D_r_1-D_c_1)*np.matrix(np.ones((D_r_1.shape[0],1))) + np.transpose(D_r_1*y)*np.matrix(np.ones((y.shape[0],1))) - np.transpose(D_c_1*y) * np.matrix(np.ones((y.shape[0],1)))
		beta_new = beta + learning_rate*delta_beta

		delta_beta2 = -1/(2*a)*b_t*(D_r_2-D_c_2)*np.matrix(np.ones((D_r_2.shape[0],1))) + np.transpose(D_r_2*y)*np.matrix(np.ones((y.shape[0],1))) - np.transpose(D_c_2*y) * np.matrix(np.ones((y.shape[0],1)))
		beta2_new = beta2 + learning_rate*delta_beta2


		delta_alpha_beta = np.linalg.norm(np.concatenate((alpha_new, beta_new, beta2_new)) -  np.concatenate((alpha, [[beta],[beta2]]))) / np.linalg.norm(np.concatenate((alpha, [[beta],[beta2]])))

		if math.isinf(delta_alpha_beta) or delta_alpha_beta < precision or np.linalg.norm(np.concatenate((delta_log_alpha, delta_beta, delta_beta2))) < precision:
			break

		alpha = np.copy(alpha_new)
		beta = beta_new[0,0]
		beta2 = beta2_new[0,0]
		#print "Iteration " + str(i)
		#print alpha
		#print beta

	return alpha, beta, beta2

def prediction(alpha, beta1, beta2, feature_filename, edge_filename, edge_filename2):
	filepath = feature_filename

	data = np.loadtxt(fname=filepath, delimiter='\t')
	data = stats.zscore(data)

	relation_filepath = edge_filename

	relation_sparse = np.loadtxt(fname=relation_filepath, delimiter='\t')

	row = relation_sparse[:,0]
	col = relation_sparse[:,1]
	weight = relation_sparse[:,2]

	R = csr_matrix((weight,(row,col))).todense()
	R = np.transpose(R)

	relation_filepath = edge_filename2

	relation_sparse = np.loadtxt(fname=relation_filepath, delimiter='\t')

	row = relation_sparse[:,0]
	col = relation_sparse[:,1]
	weight = relation_sparse[:,2]

	R2 = csr_matrix((weight,(row,col))).todense()
	R2 = np.transpose(R2)

	y = np.transpose(np.matrix(data[:,0]))
	X = np.matrix(data[:,1:])

	num_data, num_x_feature = X.shape

	D_r_1 = np.matrix(np.diag(np.squeeze(np.asarray(R.sum(axis=1)))))
	D_c_1 = np.matrix(np.diag(np.squeeze(np.asarray(R.sum(axis=0)))))

	D_r_2 = np.matrix(np.diag(np.squeeze(np.asarray(R2.sum(axis=1)))))
	D_c_2 = np.matrix(np.diag(np.squeeze(np.asarray(R2.sum(axis=0)))))


	#print type(np.transpose(alpha))
	#print type(np.ones((alpha.shape[0],1)))

	#print ((1 / (np.transpose(alpha) * np.matrix(np.ones((alpha.shape[0],1)))))[0,0])
	#print type(np.array((2*X*alpha + beta * (D_r-D_c) * np.matrix(np.ones((D_r.shape[0],1)))))[:,0])


	y_prec = (1.0 / (np.transpose(alpha) * np.matrix(np.ones((alpha.shape[0],1)))))[0,0] * (2*X*alpha + beta1 * (D_r_1-D_c_1) * np.matrix(np.ones((D_r_1.shape[0],1))) + beta2 * (D_r_2-D_c_2) * np.matrix(np.ones((D_r_2.shape[0],1))))

	return np.array(y_prec)[:,0]

def prediction_dict(alpha, beta, beta2, feature_filename, edge_filename, edge_filename2, key_filename):
	y_prec = prediction(alpha, beta, beta2, feature_filename, edge_filename, edge_filename2)

	y_prec_dict = dict()

	for line in open(key_filename):
		line_info = line.split("\t")
		key = int(line_info[0])
		username = line_info[1].split("\n")[0]
		
		y_prec_dict[username] = y_prec[key]
	
	fp = open("ccrf_result.txt", "w")
	for user in y_prec_dict:
		fp.write(user + "\t" + str(y_prec_dict[user]) + "\n")		
	fp.close()

	return y_prec_dict

def ccrf(feature_filename, edge_filename, key_filename):
	alpha, beta = learning(feature_filename, edge_filename)
	y_prec = prediction(alpha, beta, feature_filename, edge_filename)

	y_prec_dict = dict()

	for line in open(key_filename):
		line_info = line.split("\t")
		key = int(line_info[0])
		username = line_info[1].split("\n")[0]
		
		y_prec_dict[username] = y_prec[key]
		

	return y_prec_dict


