import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import nltk 
from nltk.corpus import stopwords # Stopwords 
from nltk.tokenize import word_tokenize # Word_tokenizer
from nltk.stem.porter import PorterStemmer
from pattern.text.en import singularize

from pandarallel import pandarallel

import unidecode
import re 
import time
import string
import statistics

from datetime import datetime

import contractions

stop = stopwords.words('english')

df_reviews = pd.read_json('../input/imdb-spoiler-dataset/IMDB_reviews.json', lines=True) 

print('User reviews shape: ', df_reviews.shape)

# Count missing values for each column
count = df_reviews.isnull().sum().sort_values(ascending=False)


group = df_reviews.groupby('rating') # Groupby rating
 # Get only the spoiler reviews and group them by rating
group_is_spoiler = df_reviews[df_reviews['is_spoiler'] == True].groupby('rating')

def compare_review_by_word(word,df_reviews):
    
    # Create the name for the column that will be added to the dataset
    column_name = 'contains_word_' + word 
    # Filtering reviews that contains the word "word"
    df_reviews[column_name] = df_reviews['review_text'].apply(lambda x: word.lower() in x.lower())
	
# Count the number of reviews that contains the word "word"
    n_reviews_with_word = df_reviews[df_reviews[column_name] == True].shape[0]
    # Count the number of reviews that are spoiler and contains the word "word"
    spoilers_with_word = df_reviews[(df_reviews['is_spoiler'] == True) & (df_reviews[column_name] == True)].shape[0]
    # Count the number of reviews that are NOT spoiler and contains the word "word"
    not_spoilers_with_word = df_reviews[(df_reviews['is_spoiler'] == False) & (df_reviews[column_name] == True)].shape[0]
    # Count the number of spoilers in total
    total_spoilers = df_reviews[(df_reviews['is_spoiler'] == True)].shape[0]
    
    print("Spoiler reviews that contain the word " + word + ": " + str(spoilers_with_word))
    print("Not spoiler reviews that contain the word " + word + ": " + str(not_spoilers_with_word))
    print("Total number of spoiler reviews: " + str(total_spoilers))
	
	return df_reviews
	
# Compute the number of words foreach review
df_reviews['number_of_words_in_review'] = df_reviews['review_text'].apply(lambda x: len(x.split(" ")))

# Compute the mean for spoiler and not spoiler reviews
avg_length_of_spoilers = df_reviews.groupby('is_spoiler').mean()['number_of_words_in_review']

# Print values
print('avg words for spoiler reviews: ' + str('{:.0f}'.format(avg_length_of_spoilers[1])))
print('avg words for not spoiler reviews: ' + str('{:.0f}'.format(avg_length_of_spoilers[0])))


plt.bar('Spoiler', avg_length_of_spoilers[0])
plt.bar('Not Spoiler', avg_length_of_spoilers[1])

plt.ylabel('Avg number of words per review')

plt.show()

df_reviews_dataset = df_reviews

# Remove links from text
def remove_links(text):
    """
    This function will remove links from the 
    text contained within the Dataset.
       
    arguments:
        input_text: "text" of type "String". 
                    
    return:
        value: "text" with removed links. 
    """
    
    text = re.sub(r'http\S+', '', text)
    return text

	
# Remove special characters
def removing_special_characters(text):
    """
    This Function will remove special characters (including punctualization).
    
    arguments:
         input_text: "text" of type "String".
         
    return:
        value: "text" with special characters removed.  
    """
    
    pattern = r"[^a-zA-Z]"
    text = re.sub(pattern,' ', text)
    
    return text
	
port_stemmer = PorterStemmer() # Creating the stemmer

# Apply Stemming, remove stopwords, remove meaninglessWords
def stemming_stopwords_meaninglessWords(text):
    """
    This Function will remove meaningless words, stopwords and apply stemming.
    
    arguments:
         input_text: "text" of type "String".
         
    return:
        value: "text" with meaningless and stopwords removed, stemming applied. 
    """

    tokens = word_tokenize(text) # Tokenizing the text 
    
    text_1 = ' '.join([
        port_stemmer.stem(token) for token in tokens if token not in stop and singularize(token) in words])
	text_2 = ' '.join([
        singularize(token) for token in tokens if token not in stop and singularize(token) in words
        ]) 
        
    return text_1, text_2

# Text cleaning

def text_clean(text):
    
    text = text.lower()
    
    text = remove_links(text)
    
    text = removing_special_characters(text)
    
    text_1, text_2 = stemming_stopwords_meaninglessWords(text)
    
    return text_1, text_2
	
# TRAIN SET EXCLUDING DATES < 2008 (TOO OLD) AND > 2016 (USED TO TEST)

train_set = df_reviews_dataset.loc[(df_reviews_dataset['review_date'] < '2017-01-01')
                     & (df_reviews_dataset['review_date'] > '2007-12-31')]

train_set = train_set.drop(['movie_id', 'user_id','rating', 'review_summary'],axis=1)

print("Train set shape: " + str(train_set.shape))
print("Min date: " + str(train_set['review_date'].min()))
print("Max date: " + str(train_set['review_date'].max()))

# TEST SET CONTAINING DATES > 2016

test_set = df_reviews_dataset.loc[(df_reviews_dataset['review_date'] >= '2017-01-01') 
                     & (df_reviews_dataset['review_date'] < '2018-01-01')]

test_set = test_set.drop(['movie_id', 'user_id','rating', 'review_summary'],axis=1)

print("Test set shape: " + str(test_set.shape))
print("Min date: " + str(test_set['review_date'].min()))
print("Max date: " + str(test_set['review_date'].max()))


train_set.to_csv('train_set.csv',index=False)
test_set.to_csv('test_set.csv',index=False)

df = pd.read_csv('./train_set.csv')


from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def print_kfold_scores(scores):
    print("Accuracy = ", scores['test_accuracy'])
    print("")
    print("Precision = ", scores['test_precision'])
    print("")
    print("Recall = ", scores['test_recall'])
    print("")
    print("F1 score = ", scores['test_f1_score'])
    print("")
    print("fit_time (s) = ", scores['fit_time'])
    print("")
    print("score_time (s) = ", scores['score_time'])
	
def print_cross_validation(score, stemming):
    if stemming:
        print("--------------    K-FOLD WITH STEMMING    --------------\n")
        print_kfold_scores(score)
    else:
        print("--------------    K-FOLD WITHOUT STEMMING    --------------\n")
        print_kfold_scores(score)
		
scoring = {
    'accuracy' : make_scorer(accuracy_score), 
    'precision' : make_scorer(precision_score),
    'recall' : make_scorer(recall_score), 
    'f1_score' : make_scorer(f1_score)
}


from sklearn.model_selection import StratifiedKFold

n_folds = 10
skf = StratifiedKFold(n_folds)

def apply_cross_validation(clf, X, y, ngrams = (1,1), sampling_ratio = -1, n_jobs = 5, stemming = False):
    '''
    
    '''
    
    if sampling_ratio == -1: 
        clf_pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=ngrams)),
            ('clf', clf)
        ])
    else:
        clf_pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=ngrams)),
            ('rus', RandomUnderSampler(sampling_strategy=sampling_ratio)),
            ('clf', clf)
        ])
    
    score = cross_validate(clf_pipeline, X, y, cv = skf, scoring=scoring, n_jobs = n_jobs)
    
    print_cross_validation(score, stemming)
	
	return clf_pipeline, score
	
#SVM 

# Apply cross-validation
clf_pipeline, score = apply_cross_validation(LinearSVC(), X, y)

# Register average precision and recall
avg_svm_precision = statistics.mean(score['test_precision'])
avg_svm_recall = statistics.mean(score['test_recall'])

#LogisticRegression

# Apply cross-validation
clf_pipeline, score = apply_cross_validation(LogisticRegression(max_iter=10000), X, y)

# Register average precision and recall
avg_logistic_precision = statistics.mean(score['test_precision'])
avg_logistic_recall = statistics.mean(score['test_recall'])



pd.DataFrame([
    [avg_svm_precision, avg_svm_recall],
    [avg_logistic_precision, avg_logistic_recall]
], index = ['SVM', 'Logistic'], columns=['AVG_Precision', 'AVG_Recall'])

#svm scores

def create_df_scores(scores):
    list_scores_precision = []
    list_scores_recall = []
    for x in scores:
        list_scores_precision.append(x['test_precision'])
        list_scores_recall.append(x['test_recall'])      
                                       
    score_df_precision = pd.DataFrame(list_scores_precision, index=indexes)
    score_df_recall = pd.DataFrame(list_scores_recall, index=indexes)
                                       
    score_df_precision['mean'] = score_df_precision.mean(axis=1)
    score_df_recall['mean'] = score_df_recall.mean(axis=1)
    
    score_df = pd.DataFrame([score_df_precision['mean'],score_df_recall['mean']]).T
    score_df.columns = ['avg_precision', 'avg_recall']
    
    return score_df
	
svm_df_scores = create_df_scores(svm_scores)
svm_df_scores



#lOGISTIC REGRESSION SCORES

logistic_df_scores = create_df_scores(logistic_scores)
logistic_df_scores

def get_relevant_results(df_precision, df_recall):
    
    indexes_precision = df_precision.loc[(
        df_precision > 0.45
    )].index
    
    indexes_recall = df_recall.loc[(
        df_recall > 0.45
    )].index
    
    indexes = [x for x in indexes_precision if x in indexes_recall]
    
    return indexes

logistic_indexes = get_relevant_results(logistic_df_scores['avg_precision'], logistic_df_scores['avg_recall'])
logistic_results = pd.DataFrame([logistic_df_scores['avg_precision'].rename('precision'),logistic_df_scores['avg_recall'].rename('recall')]).T.loc[logistic_indexes]
logistic_results