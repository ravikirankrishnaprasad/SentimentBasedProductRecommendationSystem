# Importing Libraries
# Importing Libraries
import pandas as pd
import re,spacy, string
import en_core_web_sm
import pickle as pk
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression

nlp = spacy.load("en_core_web_sm")

# load the pickle files 
count_vector = pk.load(open('count_vectorizer.pkl','rb'))            # Count Vectorizer
tfidf_transformer = pk.load(open('tfidf_transformer.pkl','rb')) # TFIDF Transformer

model = pk.load(open('logReg_hypar.pkl','rb'))     # Classification Model
#recommend_matrix = pk.load(open('user_final_rating.pkl','rb'))   # User-User Recommendation System 
with open('user_final_rating.pkl', 'rb') as file:
    recommend_matrix = pk.load(file)




product_df = pd.read_csv('sample30.csv',sep=",")

# Write your function here to clean the text and remove all the unnecessary elements.
def cleaning_data(text):
    text=text.lower()
    text=re.sub(r'\[.*?\]', '', text).strip() #Remove text in square brackets
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\S*\d\S*\s*','', text).strip()  # Remove words containing numbers
    return text.strip()

# Initialize stopwords from spaCy
stopword_list = nlp.Defaults.stop_words

def lemmatizer(text):
    # Tokenize and lemmatize the input text
    doc = nlp(text)
    # Remove stopwords and return lemmatized words
    sent = [token.lemma_ for token in doc if token.text.lower() not in stopword_list]
    return ' '.join(sent)


def clean_lemma(text):
    text=cleaning_data(text)
    text=lemmatizer(text)
    return text

#predicting the sentiment of the product review comments
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output

df_SR=product_df.copy()

def get_top5_user_recommendations(user):
  if user in recommend_matrix.index:
    # get the top 20  recommedation using the user_final_rating
    top20_reco = list(recommend_matrix.loc[user].sort_values(ascending=False)[0:20].index)
    
    # get the product recommedation using the orig data used for trained model
    common_top20_reco = df_SR[df_SR['id'].isin(top20_reco)]
   
    # Apply the TFIDF Vectorizer for the given 20 products to convert data in reqd format for modeling
    word_vect = count_vector.transform(common_top20_reco['reviews_text'].values.astype(str))
    X = tfidf_transformer.transform(word_vect)

    # Recommended model was LR SMOTE
    # So using the same to predict
    common_top20_reco['sentiment_pred']= model.predict(X)

    # Create a new dataframe "pred_df" to store the count of positive user sentiments
    temp_df = common_top20_reco.groupby(by='name').sum()
   
    # Create a new dataframe "pred_df" to store the count of positive user sentiments
    sent_df = temp_df[['sentiment_pred']]
    sent_df.columns = ['pos_sent_count']
   
    # Create a column to measure the total sentiment count
    sent_df['total_sent_count'] = common_top20_reco.groupby(by='name')['sentiment_pred'].count()
    
    # Calculate the positive sentiment percentage
    sent_df['pos_sent_percent'] = np.round(sent_df['pos_sent_count']/sent_df['total_sent_count']*100,2)
    
    # Return top 5 recommended products to the user
    result = sent_df.sort_values(by='pos_sent_percent', ascending=False)[:5]
    result.reset_index(inplace=True)
    return result['name']
  else:
    print(f"User name {user} doesn't exist")



