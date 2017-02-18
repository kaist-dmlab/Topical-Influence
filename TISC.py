from os import listdir
from os.path import isfile, join
import pickle
import networkx as nx
from random import randint
import generate_ccrf_feature_fast_module_pin_info_follower as gen_ccrf
import ccrf_module2 as ccrf
import sys


test_graph = pickle.load(open('graph/pinterest_test_graph.pickle'))
test_u_graph = test_graph.to_undirected()
training_graph = pickle.load(open('graph/pinterest_training_graph.pickle'))
training_u_graph = training_graph.to_undirected()

query = 'design'
feature_filename, edge_filename, edge_filename2, key_filename, regression_file_name = gen_ccrf.generateCCRFFeature(training_graph, training_u_graph, "train", query)
'''
alpha, beta, beta2 = ccrf.learning(feature_filename, edge_filename, edge_filename2)
print alpha
print beta
print beta2
'''
feature_filename, edge_filename, edge_filename2, key_filename, regression_file_name = gen_ccrf.generateCCRFFeature(test_graph, test_u_graph, "test", query)
#score_dict = ccrf.prediction_dict(alpha, beta, beta2, feature_filename, edge_filename, edge_filename2, key_filename)
'''
f = open("prediction_result.txt", 'w')
for user in score_dict:
    f.write(user + "\t" + str(score_dict[user]) + "\n")
f.close()
'''
