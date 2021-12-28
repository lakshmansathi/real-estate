import streamlit as st
import pandas as pd
import numpy
import difflib
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
import pickle
import streamlit as st
import streamlit.components.v1 as components
import base64
import streamlit as st 
import streamlit.components as stc

# Utils
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd 
df = pd.read_csv('kijiji_gta_data.csv')
X = df['description']
y = df['labelled']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# set up pipeline

# initialize
cvec = CountVectorizer(analyzer = "word",                        
                         tokenizer = None, 
                         preprocessor = None,
                         max_features = 30000,
                         ngram_range =  (1, 2))
rf = RandomForestClassifier(max_depth =  None, n_estimators =  30)
model = Pipeline([
    ('cvec', cvec),
    ('rf', rf)
])
model.fit(X_train, y_train)

user_input = st.text_input("House/Condo/bungalow")
st.header('Home App')
st.write(model.score(X_test, y_test))
prediction = model.predict([user_input])
zero = 0
one = 1
two = 2
def convert_df(df):
   return df.to_csv().encode('utf-8')
class FileDownloader(object):
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### Download File ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)
if prediction == zero:
	houses = pd.read_csv('houses_recommender.csv')
	houses = houses.iloc[: , 1:]
	houses.drop('target',axis = 1,inplace = True)
	houses.drop('labelled',axis = 1,inplace = True)
	st.markdown("#### Key Terms ####")
	houses_words = pd.read_csv('house_common_words.csv')
	st.write(houses_words)
	download = FileDownloader(houses_words.to_csv(),file_ext='csv').download()
	st.markdown("#### Quick Find ####")
	user_input_quick = st.text_input('Enter key term')
	house_list = houses[houses['description'].str.contains(user_input_quick)]
	download = FileDownloader(house_list.to_csv(),file_ext='csv').download()
	st.write(house_list)
if prediction == one:
	condos = pd.read_csv('condos_recommender.csv')
	condos = condos.iloc[: , 1:]
	condos.drop('target',axis = 1,inplace = True)
	condos.drop('labelled',axis = 1,inplace = True)
	st.markdown("#### Key Terms ####")
	condo_words = pd.read_csv('condo_common_words.csv')
	st.write(condo_words)
	download = FileDownloader(condo_words.to_csv(),file_ext='csv').download()
	st.markdown("#### Quick Find ####")
	user_input_quick = st.text_input('Enter key term')
	condo_list = condos[condos['description'].str.contains(user_input_quick)]
	download = FileDownloader(condo_list.to_csv(),file_ext='csv').download()
	st.write(condo_list)
if prediction == two:
	bungalow = pd.read_csv('bungalow_recommender.csv')
	bungalow = bungalow.iloc[: , 1:]
	bungalow.drop('target',axis = 1,inplace = True)
	bungalow.drop('labelled',axis = 1,inplace = True)
	st.markdown("#### Key Terms ####")
	bungalow_words = pd.read_csv('bungalow_common_words.csv')
	st.write(bungalow_words)
	download = FileDownloader(bungalow_words.to_csv(),file_ext='csv').download()
	st.markdown("#### Quick Find ####")
	user_input_quick = st.text_input('Enter key term')
	bungalow_list = bungalow[bungalow['description'].str.contains(user_input_quick)]
	download = FileDownloader(bungalow_list.to_csv(),file_ext='csv').download()
	st.write(bungalow_list)

st.header('Try Recommender System')
if prediction == zero:
    houses = pd.read_csv('bungalow_recommender.csv')
    houses = houses.iloc[: , 1:]
    def title_from_index(index):
    	return houses[houses.index == index]["titles"].values[0]
    def index_from_title(title):
    	title_list = houses['titles'].tolist()
    	common = difflib.get_close_matches(title, title_list, 1)
    	titlesim = common[0]
    	return houses[houses.titles == titlesim]["index"].values[0]
    	features = ['titles','locations','description']
    	for feature in features:
    		houses[feature] = houses[feature].fillna('')
    def house_combine_features(row):
    	try:
    		return row['titles'] +" "+row['locations']+row['description']
    	except:
    		st.write("Error:", row)
    houses["combined_features"] = houses.apply(house_combine_features,axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(houses["combined_features"])
    cosine_sim = cosine_similarity(count_matrix) 
    try:
    	user_houses = st.text_input("Type: house for sale (location) (anything specific)")
    except IndexError:
    	print('')
    try:
    	houses_index = index_from_title(user_houses) 
    	similar_houses =  list(enumerate(cosine_sim[houses_index]))
    	similar_houses_sorted = sorted(similar_houses,key=lambda x:x[1],reverse=True)
    	i = 1
    	for x in similar_houses_sorted:
    		index = x[0]
    		title_from_index = houses[houses['index'] == index]['titles'].values[0]
    		description_from_index = houses[houses['index'] == index]['description'].values[0]
    		website_from_index = houses[houses['index'] == index]['websites'].values[0]
    		if (i<100):
    			st.write(i,'.')
    			st.write('Title:  ',title_from_index)
    			st.write('Description:  ', description_from_index)
    			st.write('Website:  ',website_from_index)
    			i+=1
    except:
    	pass
if prediction == two:
	bungalow = pd.read_csv('bungalow_recommender.csv')
	bungalow = bungalow.iloc[: , 1:]
	def title_from_index(index):
		return bungalow[bungalow.index == index]["titles"].values[0]
	def index_from_title(title):
		title_list = bungalow['titles'].tolist()
		common = difflib.get_close_matches(title, title_list, 1)
		titlesim = common[0]
		return bungalow[bungalow.titles == titlesim]["index"].values[0]
	features = ['titles','locations','description']
	for feature in features:
		bungalow[feature] = bungalow[feature].fillna('')
	def combine_features(row):
		try:
			return row['titles'] +" "+row['locations']+row['description']
		except:
			st.write ("Error:", row)
	bungalow["combined_features"] = bungalow.apply(combine_features,axis=1)
	cv = CountVectorizer()
	count_matrix = cv.fit_transform(bungalow["combined_features"])
	cosine_sim = cosine_similarity(count_matrix)
	user_bungalow = st.text_input('Enter bungalow place to find')
	try:
		bungalow_index = index_from_title(user_bungalow)
		similar_bungalow =  list(enumerate(cosine_sim[bungalow_index]))
		similar_bungalow_sorted = sorted(similar_bungalow,key=lambda x:x[1],reverse=True)
		i = 1
		for x in similar_bungalow_sorted:
			index = x[0]
			title_from_index = bungalow[bungalow['index'] == index]['titles'].values[0]
			description_from_index = bungalow[bungalow['index'] == index]['description'].values[0]
			website_from_index = bungalow[bungalow['index'] == index]['websites'].values[0]
			if (i<100):
				st.write(i,'.',)
				st.write('Title:  ',title_from_index)
				st.write('Description:'  ,description_from_index)
				st.write('Website:  ',website_from_index)
				i +=1
	except:
		pass
	



if prediction == one:
	condos = pd.read_csv('condos_recommender.csv')
	condos = condos.iloc[: , 1:]
	def title_from_index(index):
		return condos[condos.index == index]["titles"].values[0]
	def index_from_title(title):
		title_list = condos['titles'].tolist()
		common = difflib.get_close_matches(title, title_list, 1)
		titlesim = common[0]
		return condos[condos.titles == titlesim]["index"].values[0]
	features = ['titles','locations','description']
	for feature in features:
		condos[feature] = condos[feature].fillna('')
	def combine_features(row):
		try:
			return row['titles'] +" "+row['locations']
		except:
			st.write("Error:", row)
	condos["combined_features"] = condos.apply(combine_features,axis=1)
	cv = CountVectorizer()
	count_matrix = cv.fit_transform(condos["combined_features"])
	cosine_sim = cosine_similarity(count_matrix) 
	user_condo = st.text_input('Enter condo place to find')
	try:
		condo_index = index_from_title(user_condo)
		similar_condos =  list(enumerate(cosine_sim[condo_index]))
		similar_condos_sorted = sorted(similar_condos,key=lambda x:x[1],reverse=True)
		i = 1
		for x in similar_condos_sorted:
			index = x[0]
			title_from_index = condos[condos['index'] == index]['titles'].values[0]
			description_from_index = condos[condos['index'] == index]['description'].values[0]
			website_from_index = condos[condos['index'] == index]['websites'].values[0]
			if (i<100):
				st.write(i,'.',)
				st.write('"####Title:"####  ',title_from_index)
				st.write('####Description:  ####',description_from_index)
				st.write('####Website:  ####',website_from_index)
				i+=1
	except:
		pass
	#i = 0
	#for x in similar_condos_sorted:
	#	index = x[0]
	#	title_from_index = condos[condos['index'] == index]['websites'].values[0]
	#	if (i<20):
	#		st.write(i,'.',title_from_index)
	#		i+=1











