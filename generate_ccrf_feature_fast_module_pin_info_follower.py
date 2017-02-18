import MySQLdb
from nltk.stem.porter import *
import numpy as np
import math
from scipy import stats
#import ground_truth
import pickle
import networkx as nx

SELECT_TRAINING_USER_LIST_SQL = "SELECT * FROM userList WHERE state=2"
SELECT_BOARD_BY_USER_SQL = "SELECT * FROM board WHERE user_id = %s"
SELECT_BOARD_INFO_SQL = "SELECT * FROM board WHERE board_id = %s"
SELECT_FOLLOWING_BOARD_SQL = "SELECT * FROM following_board WHERE user_id = %s and board_id = %s"
SELECT_FOLLOWING_USER_SQL = "SELECT * FROM following_user WHERE user_id = %s and following_id = %s"
SELECT_COUNT_PIN_SQL = "SELECT count(*) FROM pin WHERE board_href = %s"
SELECT_USER_LIST_SQL = "SELECT * FROM userInfo ORDER BY follower_cnt DESC LIMIT 5000"
SELECT_GROUND_TRUTH_INFO_SQL = "SELECT * FROM ground_truth_info WHERE username = %s"

def connectDB(db_name):
	db = None
	cursor = None

	if db_name == 'pin':
		db = MySQLdb.connect(host='dmserver1.kaist.ac.kr', user='daehoon', passwd='rlaeogns', db='pinterest_design_pin', charset='utf8', use_unicode=True)
		cursor = db.cursor()
		cursor.execute("set names utf8")

	elif db_name == 'all':
		db = MySQLdb.connect(host='dmserver1.kaist.ac.kr', user='daehoon', passwd='rlaeogns', db='pinterest_design', charset='utf8', use_unicode=True)
		cursor = db.cursor()
	  	cursor.execute("set names utf8")

	return db, cursor

def closeDB(db, cur):
	cur.close()
	db.close()

def selectDB(db, cursor, SqlQuery, params=()):
	cursor.execute(SqlQuery, params)
	return cursor.fetchall()

def getUserOfBoard(u_graph, board_href):
	neighbors = u_graph[board_href]
	for neighbor in neighbors:
		if neighbors[neighbor]['type'] == 'curated':
			return neighbor

def getBoardList(u_graph, username):
	board_list = list()
	neighbors = u_graph[username]
	for neighbor in neighbors:
		if neighbors[neighbor]['type'] == 'curated':
			board_list.append(neighbor)
	return board_list

def getBoardFollowerList(u_graph, board_id):
	follower_list = list()
	neighbors = u_graph[board_id]
	for neighbor in neighbors:
		if neighbors[neighbor]['type'] == 'following':
			follower_list.append(neighbor)
	return follower_list

def reverseScoreFeature(feature):
	if np.isnan(feature):
		feature = 0.0
	elif feature == 0.0:
		feature = 1.0
	else:
		feature = 1.0 / feature

	return feature

def isFollowingBoard(graph, user_id, board_id):
	if board_id in graph[user_id]:
		if graph[user_id][board_id]['type'] == 'following':
			return True
	return False

def getGroundTruthFeatureByFile(filepath):
	user_ground_truth_feature = dict()
	f = open(filepath)
	idx = 0
	for line in f:
		if idx > 0:
			feature_info = line.split('\t')
			username = feature_info[0]
			user_ground_truth_feature[username] = dict()
			user_ground_truth_feature[username]['like'] = int(feature_info[1])
			user_ground_truth_feature[username]['repin'] = int(feature_info[2])
			user_ground_truth_feature[username]['comment'] = int(feature_info[3])
			user_ground_truth_feature[username]['word'] = int(feature_info[4])
			user_ground_truth_feature[username]['query_follower'] = int(feature_info[5])
			user_ground_truth_feature[username]['query_follower_weight'] = int(feature_info[6])
		idx += 1

	return user_ground_truth_feature

def getGroundTruthFeatureInDB(db, cur, user_list):
	user_ground_truth_feature = dict()
	for username in user_list:
		result = selectDB(db, cur, SELECT_GROUND_TRUTH_INFO_SQL, params=(username,))

		if len(result) > 0:
			ground_truth_info = result[0]
			comment_cnt = ground_truth_info[2]
			word_cnt = ground_truth_info[3]
			like_cnt = ground_truth_info[4]
			repin_cnt = ground_truth_info[5]
			query_follower = ground_truth_info[6]
			query_follower_weight = ground_truth_info[7]
		else:
			comment_cnt = 0
			word_cnt = 0
			like_cnt = 0
			repin_cnt = 0
			query_follower = 0
			query_follower_weight = 0

		user_ground_truth_feature[username] = dict()
		user_ground_truth_feature[username]['like'] = like_cnt
		user_ground_truth_feature[username]['repin'] = repin_cnt
		user_ground_truth_feature[username]['comment'] = comment_cnt
		user_ground_truth_feature[username]['word'] = word_cnt
		user_ground_truth_feature[username]['query_follower'] = query_follower
		user_ground_truth_feature[username]['query_follower_weight'] = query_follower_weight

	return  user_ground_truth_feature

def convertZscore(score_dict):
	repin_score = list()
	repin_zscore = list()
	word_score = list()
	word_zscore = list()
	like_score = list()
	like_zscore = list()
	comment_score = list()
	comment_zscore = list()
	query_follower_score = list()
	query_follower_zscore = list()
	query_follower_weight_score = list()
	query_follower_Weight_zscore = list()

	user_key_list = list()

	feature_zscore = dict()

	for key in score_dict:
		repin = score_dict[key]['repin']
		word = score_dict[key]['word']
		like = score_dict[key]['like']
		comment = score_dict[key]['comment']
		query_follower = score_dict[key]['query_follower']
		query_follower_weight = score_dict[key]['query_follower_weight']

		repin_score.append(repin)
		word_score.append(word)
		like_score.append(like)
		comment_score.append(comment)
		query_follower_score.append(query_follower)
		query_follower_weight_score.append(query_follower_weight)

		user_key_list.append(key)

	repin_zscore = stats.zscore(np.array(repin_score))
	word_zscore = stats.zscore(np.array(word_score))
	like_zscore = stats.zscore(np.array(like_score))
	comment_zscore = stats.zscore(np.array(comment_score))
	query_follower_zscore = stats.zscore(np.array(query_follower_score))
	query_follower_weight_zscore = stats.zscore(np.array(query_follower_weight_score))

	for idx, user in enumerate(user_key_list):
		feature_zscore[user] = dict()
		feature_zscore[user]['repin'] = repin_zscore[idx]
		feature_zscore[user]['word'] = word_zscore[idx]
		feature_zscore[user]['like'] = like_zscore[idx]
		feature_zscore[user]['comment'] = comment_zscore[idx]
		feature_zscore[user]['query_follower'] = query_follower_zscore[idx]
		feature_zscore[user]['query_follower_weight'] = query_follower_weight_zscore[idx]

	return feature_zscore


def convertDictToZscore(data_dict):
	z_score_dict = dict()
	score_list = list()
	key_list = list()
	for key in data_dict.keys():
		score_list.append(data_dict[key])
		key_list.append(key)

	z_score = stats.zscore(np.array(score_list))

	for idx, key in enumerate(key_list):
		z_score_dict[key] = z_score[idx]

	return z_score_dict


def calculateGroundTruth(feature_list):
	result = 0.0
	for feature in feature_list:
		result += feature

	''' PRODUCT
	result = 1.0
	for feature in feature_list:
		result *= feature
	'''
	''' MEAN
	result = np.mean(feature_list)
	'''

	return result


def getNumOfPinGraph(u_graph, board_id):
	pin_num = 0

	if 'pin_num' in u_graph.node[board_id]:
		pin_num = u_graph.node[board_id]['pin_num']

	return pin_num

	'''

	pin_list = list()
	neighbors = u_graph[board_id]
	for neighbor in neighbors:
		if neighbors[neighbor]['type'] == 'curated-pin':
			pin_list.append(neighbor)

	return len(pin_list)
	'''

###############  MAIN LOGIC  ###############

def generateCCRFFeature(graph, u_graph, file_postfix, query):

	#file_postfix = '20150914_following'
	#query = 'design'
	#graph = pickle.load(open('../Graph/query_connected_graph_pin_info.pickle'))
	#u_graph = graph.to_undirected()

	# Get target user list
	user_list = list()
	for node in graph.nodes():
		if graph.node[node]['type'] == 'user':
			user_list.append(node)

	print "User Cnt : " + str(len(user_list))

	#query = 'design'

	pin_db, pin_cur = connectDB('pin')
	all_db, all_cur = connectDB('all')


	user_ground_truth_feature = getGroundTruthFeatureInDB(pin_db, pin_cur, user_list)
	#print user_ground_truth_feature
	#user_ground_truth_feature = getGroundTruthFeatureByFile('ground_truth_feature_training_' + file_postfix + '.txt')
	user_ground_truth_feature_z_score = convertZscore(user_ground_truth_feature)

	#user_list = user_ground_truth_feature.keys()
	#print "Valid User Cnt : " + str(len(user_list))

	feature_file_name = 'pinterest_train_' + file_postfix + '.txt'
	regression_file_name = 'regression_train_' + file_postfix + '.txt'

	# just for information
	fp = open('user_feature_train_'+file_postfix+'.txt', 'w')
	fp3 = open('all_feature_train_'+file_postfix+'.txt', 'w')

	# return feature file
	fp2 = open(feature_file_name, 'w')
	fp_regression = open(regression_file_name, 'w')

	user_feature = dict()

	print "Start generating features"

	user_ground_truth_feature_follower = dict()

	for idx, username in enumerate(user_list):
		#username = result[0]
		#if idx % 100 == 0:
		#	print str(idx)

		if not user_feature.has_key(username):
			user_feature[username] = dict()
			user_feature[username]['pin'] = list()
			user_feature[username]['follower'] = list()
			user_feature[username]['all_pin'] = list()
			user_feature[username]['unique_follower'] = set()
			user_feature[username]['all_follower'] = list()

		board_list = getBoardList(u_graph, username)

		for board in board_list:
			board_href = board
			board_category = graph.node[board]['category']
			#board_info = getBoardInfoDB(all_db, all_cur, board_href)
			#pin_num = board_info[4] ## TODO : Change get pins on graph
			pin_num  = getNumOfPinGraph(u_graph, board_href)
			follower_list = getBoardFollowerList(u_graph, board_href)
			follower_num = len(follower_list)

			if board_category == query:
				#print str(getNumOfCrawledPin(pin_db, pin_cur, board_href)) + " / " + str(pin_num)
				#pin_num = getNumOfCrawledPin(pin_db, pin_cur, board_href)
				user_feature[username]['pin'].append(pin_num)
				user_feature[username]['follower'].append(follower_num)
				for follower in follower_list:
					user_feature[username]['unique_follower'].add(follower)

			user_feature[username]['all_pin'].append(pin_num)
			user_feature[username]['all_follower'].append(follower_num)

		follower_feature = sum(user_feature[username]['follower'])

		user_ground_truth_feature_follower[username] = follower_feature

	user_ground_truth_feature_follower_zscore = convertDictToZscore(user_ground_truth_feature_follower)

	for idx, username in enumerate(user_list):
		#### Generate Feature ####
		#### Feature 1 ####
		feature1 = sum(user_feature[username]['pin'])
		#### Feature 2 ####
		feature2 = sum(user_feature[username]['follower'])
		#### feature 3 ####
		feature3 = np.inner(user_feature[username]['pin'], user_feature[username]['follower'])
		if feature3 == False:
			feature3 = 0.0
		#### feature 4 ####
		feature4 = 0.0
		all_pin_num = sum(user_feature[username]['all_pin'])
		query_pin_num = sum(user_feature[username]['pin'])
		if all_pin_num > 0:
			feature4 = query_pin_num / float(all_pin_num)

		#### feature 5 ####

		feature_follow_ratio = 0.0
		all_follow_num = sum(user_feature[username]['all_follower'])
		follow_num = sum(user_feature[username]['follower'])
		if all_follow_num > 0:
			feature_follow_ratio = follow_num / float(all_follow_num)

		### feature unique follower ##
		feature_u_f = 0
		if username in user_feature:
			feature_u_f = len(user_feature[username]['unique_follower'])


		### feature 5 ###
		#feature5 = 0.0
		#feature5 = reverseScoreFeature(np.std(user_feature[username]['pin']))
		### feature 6 ###
		#feature6 = 0.0
		#feature6 = reverseScoreFeature(np.std(user_feature[username]['follower']))
		### feature 7 ###
		#feature7 = 0.0
		#pinBYfollower = list()
		#for idx in range(0, len(user_feature[username]['pin'])):
		#	pinBYfollower.append(user_feature[username]['pin'][idx] * user_feature[username]['follower'][idx])
		#feature7 = reverseScoreFeature(np.std(pinBYfollower))


		result_content = username + "\t" + str(feature1) + "\t" + str(feature2) + "\t" +  str(feature3) + "\t" + str(feature4) + "\t" + str(feature_u_f)# + "\t" + str(feature5) + "\t" + str(feature6) + "\t" + str(feature7)
		fp.write(result_content + "\n")

		#score = calculateGroundTruth([user_ground_truth_feature_z_score[username]['repin'], user_ground_truth_feature_z_score[username]['like'], user_ground_truth_feature_z_score[username]['comment'], user_ground_truth_feature_z_score[username]['query_follower'], user_ground_truth_feature_z_score[username]['word'], user_ground_truth_feature_z_score[username]['query_follower_weight']])

		#score = calculateGroundTruth([user_ground_truth_feature_z_score[username]['repin'], user_ground_truth_feature_z_score[username]['like'], user_ground_truth_feature_z_score[username]['comment'], user_ground_truth_feature_z_score[username]['word']])

		score = calculateGroundTruth([user_ground_truth_feature_z_score[username]['repin'], user_ground_truth_feature_z_score[username]['word'], user_ground_truth_feature_follower_zscore[username]])

		result_content = str(score) + "\t" + str(feature1) + "\t" + str(feature4) + "\t" + str(feature_follow_ratio) # str(feature2) + "\t" +  str(feature3) + "\t" + str(feature4) + "\t" + str(feature_u_f)# + "\t" + str(feature5) + "\t" + str(feature6) + "\t" + str(feature7)
		fp2.write(result_content + "\n")

		fp_regression.write(str(score) + "\t" + str(feature1) + "\t" + str(feature2) + "\t" + str(feature3) + "\t" + str(feature4) + "\t" + str(feature_follow_ratio) + "\n")

		y_feature1 = user_ground_truth_feature_z_score[username]['repin']
		y_feature2 = user_ground_truth_feature_z_score[username]['like']
		y_feature3 = user_ground_truth_feature_z_score[username]['word']
		y_feature4 = user_ground_truth_feature_z_score[username]['comment']
		y_feature5 = user_ground_truth_feature_z_score[username]['query_follower']
		y_feature6 = user_ground_truth_feature_z_score[username]['query_follower_weight']
		y_feature7 = user_ground_truth_feature_follower_zscore[username]

		result_content = str(y_feature1) + "\t" +  str(y_feature2) + "\t" +  str(y_feature3) + "\t" + str(y_feature4) + "\t" + str(y_feature5) + "\t" + str(y_feature6)+ "\t" + str(y_feature7) + "\t" + str(feature1) + "\t" + str(feature2) + "\t" +  str(feature3) + "\t" + str(feature4) + "\t" + str(feature_u_f)# + "\t" + str(feature5) + "\t" + str(feature6) + "\t" + str(feature7)

		#print result_content
		fp3.write(result_content + "\n")

	fp.close()
	fp2.close()
	fp3.close()
	fp_regression.close()

	print "End generating features"


	print "Start generating edge features"

	### feature 8 ###

	edge_file_name = 'edge_list_key_train_'+file_postfix+'.txt'
	edge_file_name2 = 'edge2_list_key_train_'+file_postfix+'.txt'
	key_file_name = 'edge_key_train_'+file_postfix+'.txt'

	fp = open('edge_feature_train_'+file_postfix+'.txt','w')
	fp2 = open('edge_list_train_'+file_postfix+'.txt','w')
	fp3 = open(key_file_name,'w')
	fp4 = open(edge_file_name,'w')
	fp5 = open(edge_file_name2, 'w')

	## TODO USER TO KEY ##

	user_key = dict()
	following_dict = dict()


	user_board_size = dict()
	user_all_board_size = dict()

	for idx, username in enumerate(user_list):
		user_key[username] = idx
		fp3.write(str(user_key[username]) + "\t" + username + "\n")

		following_dict[username] = dict()
		following_dict[username]['all_following_pin'] = 0.0
		user_board_size[username] = 0.0

		board_list = getBoardList(u_graph, username)
		for board in board_list:
			#board_info = getBoardInfoDB(all_db, all_cur, board)
			#pin_num = board_info[4]
			pin_num  =  getNumOfPinGraph(u_graph, board)
			board_category = graph.node[board]['category']
			if board_category == query:
				user_board_size[username] += pin_num

			if not username in user_all_board_size:
				user_all_board_size[username] = 0
			user_all_board_size[username] += pin_num

	## GET BOARD FOLLOWING EDGE ##

	for idx, edge in enumerate(graph.edges()):
		if graph[edge[0]][edge[1]]['type'] == 'following':
			board_href = edge[1]
			board_category = graph.node[board_href]['category']

			if board_category == query:
				#board_info = getBoardInfoDB(all_db, all_cur, board_href)
				#board_user = board_info[6].split('/')[1]
				board_user = getUserOfBoard(u_graph, board_href)
				#board_user = graph.node[board_href]['user']
				#board_user = edge[1].split('/')[1]
				if not board_user in following_dict[edge[0]]:
					following_dict[edge[0]][board_user] = dict()
					following_dict[edge[0]][board_user]['following'] = 0.0
					following_dict[edge[0]][board_user]['all'] = 0.0

				#pin_num = board_info[4] ## TODO : Change get pins on graph
				pin_num  = getNumOfPinGraph(u_graph, board_href)
				following_dict[edge[0]][board_user]['following'] += pin_num
				following_dict[edge[0]]['all_following_pin'] += pin_num

	## edge[user][user]['follwoing']
	## edge[user][user]['all']
	## Calculating Score ##

	following_ratio_dict = dict()
	for user in following_dict:
		following_ratio_dict[user] = list()
		for following_user in following_dict[user]:
			if following_user != "all_following_pin":
				query_following_pin_cnt = following_dict[user][following_user]['following']
				query_all_pin_cnt = user_board_size[following_user] ###
				if query_all_pin_cnt > 0.0:
					ratio = float(query_following_pin_cnt) / query_all_pin_cnt
					following_ratio_dict[user].append(ratio)


	for user in following_dict:
		for following_user in following_dict[user]:
			if following_user != 'all_following_pin':

				query_following_pin_cnt = following_dict[user][following_user]['following']
				query_all_pin_cnt = user_board_size[following_user] #user_board_size[following_user]
				feature8 = 0.0
				sum_following_ratio = 0.0

				if len(following_ratio_dict) > 0:
					sum_following_ratio = sum(following_ratio_dict[user])

				if query_all_pin_cnt > 0.0 and sum_following_ratio > 0:
					feature8 = (float(query_following_pin_cnt) / query_all_pin_cnt) / float(sum_following_ratio)
					#print user + "\t" + following_user + "\t" + str(feature8)
				#else:
					#print "Pin Zero User :: " + following_user

				'''
				#print str(user) + "\t" + str(following_user)
				query_following_pin_cnt = following_dict[user][following_user]['following']
				feature8 = 0.0
				if following_dict[user]['all_following_pin'] > 0:
					feature8 = float(query_following_pin_cnt) / following_dict[user]['all_following_pin']
				'''

				#print user + "\t" + following_user + "\t" + str(feature8)
				#print str(user_key[user]) + "\t" + str(user_key[following_user]) + "\t" + str(feature8)
				if feature8 > 0.0:
					fp2.write(user + "\t" + following_user + "\t" + str(feature8) + "\n")
					fp4.write(str(user_key[user]) + "\t" + str(user_key[following_user]) + "\t" + str(feature8) + "\n")



				query_following_pin_cnt = following_dict[user][following_user]['following']
                                feature9 = 0.0

				if following_dict[user]['all_following_pin'] > 0:
                                        feature9 = float(query_following_pin_cnt) / following_dict[user]['all_following_pin']

                                #print user + "\t" + following_user + "\t" + str(feature8)
                                #print str(user_key[user]) + "\t" + str(user_key[following_user]) + "\t" + str(feature8)
                                if feature9 > 0.0:
                                        fp5.write(str(user_key[user]) + "\t" + str(user_key[following_user]) + "\t" + str(feature9) + "\n")



	## Sparse Matrix Size Setting ##
	max_user_key = len(user_key) - 1
	fp4.write("0\t0\t0.0\n")
	fp4.write(str(max_user_key) + "\t" + str(max_user_key) + "\t" + str(0.0) + "\n")
	fp5.write("0\t0\t0.0\n")
	fp5.write(str(max_user_key) + "\t" + str(max_user_key) + "\t" + str(0.0) + "\n")


	fp.close()
	fp2.close()
	fp3.close()
	fp4.close()
	fp5.close()

	print "End generating edge features"

	closeDB(pin_db, pin_cur)
	closeDB(all_db, all_cur)

	return feature_file_name, edge_file_name, edge_file_name2, key_file_name, regression_file_name

#generateCCRFFeature()
