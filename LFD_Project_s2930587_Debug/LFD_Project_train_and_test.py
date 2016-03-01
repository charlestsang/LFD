
#from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, homogeneity_completeness_v_measure#
#from sklearn import cluster
#from sklearn.cluster import KMeans
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cluster
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from nltk.tokenize import *
import xml.etree.ElementTree as ET
import glob
import os 
import sklearn
import nltk
import numpy as np
import sys
import time
import re
import string
import itertools

start_time = time.time() 


# read both training / testing data based on language: dutch, english, italian, spanish
def read_data(lang, folder):
	documents = []
	users = []
	genders = []
	ages = []
	Dict = defaultdict(list)

	if folder == 'training':
		with open(folder +'/'+lang +'/'+'truth.txt', encoding='utf-8') as f:
			for line in f:
				tokens = line.strip().split(':::')
				# taokens[0] = user_id, tokens[1] = gender, tokens[2] = age
				Dict[tokens[0]] = (tokens[1], tokens[2])   
	
	#reading all XML files and make into list
	path = folder +'/'+lang+'/'
	for XMLinfile in glob.glob( os.path.join(path,'*.xml')):
		xmlD = ET.parse(XMLinfile)
		root = xmlD.getroot()
		for child in root:
			tweet_info = []
			tweet_info.append(child.text.split())
			documents.append(tweet_info)
			users.append(root.attrib['id'])
			
			if folder == 'training':
				genders.append(Dict[root.attrib['id']][0])
				ages.append(Dict[root.attrib['id']][1])
	if folder == 'training':
		return documents, users, genders, ages
	else:
		return users, documents


def identity(x):
	return x

def tokenizer(document):
	#tknzr = TweetTokenizer()
	#tknzr.tokenize(document)
	#return ' '.join(tknzr.tokenize(document))
	return ' '.join(nltk.word_tokenize(document))

def preprocessing(document):
	document = list(document[0])
	newDoc = []
	#punctuation = {'!','#', ',', '.', '(',')', '?', '/', '$', '\\', '|', '^', '*', '&'}

	for word in document:
		if word.startswith('#'):
			newDoc.append('#hashtag')
		elif word.startswith('http'):
			newDoc.append('url')
		elif word.startswith('@'):
			newDoc.append('at')
		elif word.startswith('www\.'):
			newDoc.append('url')
		elif word.startswith('&'):
			newDoc.append('and')
		elif word.startswith("\bdon't\b"):
			newDoc.append('do not')
		elif word.startswith("\bdidn't\b"):
			newDoc.append('did not')	
		elif word.startswith("\bhaven't\b"):
			newDoc.append('have not')
		elif word.startswith("\bwon't\b"):
			newDoc.append('will not')
		elif word.startswith("\bhasn't\b"):
			newDoc.append('has not')
		elif word.startswith("\bwouldn't\b"):
			newDoc.append('would not')
		elif word.startswith("\bcan't\b"):
			newDoc.append('can not')
		else:
			newDoc.append(word)
	#remove punctuation
	#newDoc =''.join(char for char in document if char in string.punctuation)
	newDoc =' '.join(newDoc)
	return newDoc

def ageClass(doc, age):

	if len(sys.argv) == 3:
		split_point = int(0.75*len(doc))  
		Xtrain = doc[:split_point]  
		Ytrain_age = age[:split_point]             
		Xtest = doc[split_point:]
		Ytest_age = age[split_point:]          

	elif len(sys.argv) == 4:
		Xtrain= doc   
		Ytrain_age = age 
	  
	tfidf = True
	if tfidf:
		vec = TfidfVectorizer(preprocessor = preprocessing, tokenizer = tokenizer, sublinear_tf=True, max_df= 0.5) #min_df =1
	else:
	    vec = CountVectorizer(preprocessor = preprocessing, tokenizer = tokenizer)

	if len(sys.argv) == 3:
		if sys.argv[2] == 'spanish': 
			cls = svm.SVC(kernel='rbf', gamma=1.0, C=4.5)        #slow, but accuracy higher than NB, not applicable for Dutch/Italian
		elif sys.argv[2] == 'english':
			cls = svm.SVC(kernel='linear', C = 5)              #slow, but accuracy higher than NB, not applicable for Dutch/Italian
			#cls = svm.SVC(kernel='rbf', gamma=1.0, C=5)
			#cls = MultinomialNB()
		elif sys.argv[2] == 'dutch' or sys.argv[2] == 'italian':
			cls = MultinomialNB()                                #fastest, but accuracy low
		
		classifier = Pipeline( [('vec', vec), ('cls', cls)] )
		classifier.fit(Xtrain, Ytrain_age)
		Yguess_age = classifier.predict(Xtest)
		
		print ("Accuracy_Age:", accuracy_score(Ytest_age, Yguess_age))
		print (classification_report(Ytest_age, Yguess_age))
		print ("Confusion_Matrix_Age:", confusion_matrix(Ytest_age, Yguess_age))

	else:
		if sys.argv[3] == 'spanish': 
			cls = svm.SVC(kernel='rbf', gamma=1.0, C=4.5)        
			#cls = svm.SVC(kernel='linear', C = 0.8)             
		elif sys.argv[3] == 'english':
			#cls = svm.SVC(kernel='linear', C = 0.1)             
			cls = svm.SVC(kernel='rbf', gamma=1.0, C=5)
			#cls = MultinomialNB()		
		elif sys.argv[3] == 'dutch' or sys.argv[3] == 'italian':
			cls = MultinomialNB()
		
		classifier = Pipeline( [('vec', vec), ('cls', cls)] )
		classifier.fit(Xtrain, Ytrain_age)

		return classifier

def genderClass(doc, gender):

	if len(sys.argv) == 3:
		split_point = int(0.75*len(doc))  
		Xtrain = doc[:split_point]      
		Ytrain_gender = gender[:split_point]        
		Xtest = doc[split_point:]          
		Ytest_gender = gender[split_point:]

	elif len(sys.argv) == 4:

		split_point = int(0.75*len(doc))
		Xtrain= doc   
		Ytrain_gender = gender   
    
	tfidf = True
	if tfidf:
		vec = TfidfVectorizer(preprocessor = preprocessing, tokenizer = tokenizer, sublinear_tf=True, max_df= 0.5) #min_df =1
	else:
	    vec = CountVectorizer(preprocessor = preprocessing, tokenizer = tokenizer)

	# combine the vectorizer with a classifier
	cls = svm.SVC(kernel='rbf', gamma=1.0, C=4.5)   #slow, but accuracy higher than NB
	#cls = svm.SVC(kernel='linear', C = 5)        #slow, but accuracy higher than NB
	#cls = MultinomialNB()                          #fastest, but accuracy low
	classifier = Pipeline( [('vec', vec), ('cls', cls)] )
	classifier.fit(Xtrain, Ytrain_gender)
	
	if len(sys.argv) == 3:
		Yguess_gender = classifier.predict(Xtest)
		print ("Accuracy_Gender:", accuracy_score(Ytest_gender, Yguess_gender))
		print (classification_report(Ytest_gender, Yguess_gender))
		print ("Confusion_Matrix_Gender:", confusion_matrix(Ytest_gender, Yguess_gender))

	else:
		return classifier

def Prediction(Yguess_gender, Yguess_age, user_test, doc_test):
	testlist = []
	#a = [user_test, doc_test]
	#testlist = list(itertools.chain(*a))
	for user, doc in zip(user_test, doc_test):
		testlist.append([user,doc])

	for doc_list in testlist:
		doc_list.append(Yguess_gender.predict([doc_list[1]]))

		if Yguess_age == 'dut_age_Classifier':
			doc_list.append('XX-XX')
		elif Yguess_age == 'ita_age_Classifier':
			doc_list.append('XX-XX')
		else:
			doc_list.append(Yguess_age.predict([doc_list[1]]))
	
	return testlist
	#print (testlist)

def output_truth_file(lang, testlist):
	filename = 'testing/' + lang + '/truth.txt'
	truthfile = open(filename, 'w')
	genDict = defaultdict(list)
	ageDict = defaultdict(list)
	#DefDict = defaultdict(list)
	
	if lang == 'english':
		for truthlist in testlist:
			genDict[truthlist[0]].append(truthlist[2][0])
			ageDict[truthlist[0]].append(truthlist[3][0])
		for key, value in genDict.items():
			truthfile.write(str(key) + ':::' + str(Counter(value).most_common(1)[0][0]) + ':::' +str(Counter(ageDict[key]).most_common(1)[0][0])+ '\n')
	
	elif lang == 'spanish':
		for truthlist in testlist:
			genDict[truthlist[0]].append(truthlist[2][0])
			ageDict[truthlist[0]].append(truthlist[3][0])	
		for key, value in genDict.items():
			truthfile.write(str(key) + ':::' + str(Counter(value).most_common(1)[0][0]) + ':::' +str(Counter(ageDict[key]).most_common(1)[0][0])+'\n')
	
	else:
		for truthlist in testlist:
			genDict[truthlist[0]].append(truthlist[2][0])
		for key, value in genDict.items():
			truthfile.write(str(key) + ':::' + str(Counter(value).most_common(1)[0][0]) + ':::' + 'XX-XX' + '\n')

if __name__ == '__main__':

    #if training input <xxxx.py> <training> <language>

	if len(sys.argv) == 3:
		folder = sys.argv[1].lower()
		lang = sys.argv[2].lower()

		print('Running training file...make sure input correct language!',file=sys.stderr)
		
		if lang == 'dutch':
			D_document, D_user, D_gender, D_age = read_data(lang, folder)
			genderClass(D_document, D_gender)
			ageClass(D_document, D_age)
		elif lang == 'english':
			E_document, E_user, E_gender, E_age = read_data(lang, folder)
			genderClass(E_document, E_gender)
			ageClass(E_document, E_age)
		elif lang =='italian':
			I_document, I_user, I_gender, I_age = read_data(lang, folder)
			genderClass(I_document, I_gender)
			ageClass(I_document, I_age)
		elif lang =='spanish':
			S_document, S_user, S_gender, S_age = read_data(lang, folder)
			genderClass(S_document, S_gender)
			ageClass(S_document, S_age)

	
	# if testing: input: <test.py> <training> <texting> <lang>
	
	elif len(sys.argv) == 4:
		folder_1 = sys.argv[1].lower()
		folder_2 = sys.argv[2].lower()
		lang = sys.argv[3].lower()
		
		print('Caution: Input_format: <XX.py> <training> <testing> <language>')
		print('Running testing files...', file=sys.stderr)	
		if lang == 'dutch':
			D_document, D_user, D_gender, D_age= read_data(lang, folder_1)
			doc_test_dut, user_test_dut = read_data(lang, folder_2)
			dut_gender_Classifier = genderClass(D_document, D_gender)
			dut_age_Classifier = ageClass(D_document, D_age)
			dut_Prediction = Prediction(dut_gender_Classifier, dut_age_Classifier, doc_test_dut, user_test_dut)
			output_truth_file(lang, dut_Prediction)

		elif lang == 'english':
			E_document, E_user, E_gender, E_age= read_data(lang, folder_1)
			doc_test_eng, user_test_eng = read_data(lang, folder_2)
			eng_gender_Classifier = genderClass(E_document, E_gender)
			eng_age_Classifier = ageClass(E_document, E_age)
			eng_predictions = Prediction(eng_gender_Classifier, eng_age_Classifier, doc_test_eng, user_test_eng)
			output_truth_file(lang, eng_predictions)

		elif lang == 'italian':
			I_document, I_user, I_gender, I_age= read_data(lang, folder_1)
			doc_test_ita, user_test_ita = read_data(lang, folder_2)
			ita_gender_Classifier = genderClass(I_document, I_gender)
			ita_age_Classifier = ageClass(I_document, I_age)
			ita_predictions = Prediction(ita_gender_Classifier, ita_age_Classifier, doc_test_ita, user_test_ita)
			output_truth_file(lang, ita_predictions)

		elif lang == 'spanish':
			S_document, S_user, S_gender, S_age= read_data(lang, folder_1)
			doc_test_spa, user_test_spa = read_data(lang, folder_2)
			spa_gender_Classifier = genderClass(S_document, S_gender)
			spa_age_Classifier = ageClass(S_document, S_age)
			spa_predictions = Prediction(spa_gender_Classifier, spa_age_Classifier, doc_test_spa, user_test_spa)
			output_truth_file(lang, spa_predictions)

		print("Writing truth files: Finished!")

print ("time spent:", time.time() - start_time)
