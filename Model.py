###Importing dependencies

import numpy as np
import pandas as pd       
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
# train.shape should be (25000,3)


test = pd.read_csv("testData.tsv", header=0, \
                    delimiter="\t", quoting=3)

import pickle
import requests
import json

##Cleaning the data
import nltk
nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import packages

import bs4 as bs
import nltk
from nltk.tokenize import sent_tokenize # tokenizes sentences
import re
from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

###clean the review

def review_cleaner(review):
    '''
    Clean and preprocess a review.
    
    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    '''
    
    #1. Remove HTML tags
    review = bs.BeautifulSoup(review).text
    
    #2. Use regex to find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)
    
    #3. Remove punctuation
    review = re.sub("[^a-zA-Z]", " ",review)
    
    #4. Tokenize into words (all lower case)
    review = review.lower().split()
    
    #5. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    review = [w for w in review if not w in eng_stopwords]
    
    #6. Join the review to one sentence
    review = ' '.join(review+emoticons)
    # add emoticons to the end

    return(review)



num_reviews = len(train['review'])

review_clean_original = []

for i in range(0,num_reviews):
    if( (i+1)%500 == 0 ):
        # print progress
        print("Done with %d reviews" %(i+1)) 
    review_clean_original.append(review_cleaner(train['review'][i]))

####Sentiment classification
## Example code BoW

from sklearn.feature_extraction.text import CountVectorizer
# or from sklearn.feature_extraction.text import TfidfVectorizer

sent1 = "cool students study cool data science"
sent2 = "to know data science study data science"

vect = CountVectorizer() #instantiate
#vect2 = TfidfVectorizer()

sents = np.array([sent1,sent2])

vect.fit(sents);

print('Total number of words in the vocabulary (and position in feature matrix):\n')
print(vect.vocabulary_)
# vocabulary for the BoW model is stored in a dictionary


# Transform to get feature vectors

bag = vect.transform(sents)

bag.toarray()

# the rows corresponds to the sentences 

vect.get_feature_names() # stored in the right places

# Put it in a DataFrame for interpretability

pd.DataFrame(bag.toarray(), columns=vect.get_feature_names(), index=[sent1,sent2])

# the number in the DataFrame is called Raw Term frequency raw term frequencies: 
# tf (t,d)—the number of times a term t occurs in a document d.

##classify IMDB review dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(\
    review_clean_original, train['sentiment'], random_state=0, test_size=.2)


# CountVectorizer can actucally handle a lot of the preprocessing for us
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# Transform the text data to feature
# Only fit training data (to mimic real world)

vectorizer.fit(X_train)

# Check that it worked, 
# now we have fitted a model that can transform features
# to sparse matrix representation

print(vectorizer.get_feature_names()[:10])

train_bag = vectorizer.transform(X_train) #transform to a feature matrix
test_bag = vectorizer.transform(X_test)

print(train_bag.toarray().shape) # 20,000 reviews, 2,000 feartures. just as expected
print(test_bag.toarray().shape)

type(train_bag) # sparse matrix representation

print(train_bag)

##Classify with randomforest
from sklearn.ensemble import RandomForestClassifier

## Initialize a Random Forest classifier with 50 trees
# hyperparameter n_estimators always set in instantiation

forest = RandomForestClassifier(n_estimators = 50) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the target variable

forest = forest.fit(train_bag, y_train) # can take 20 seconds to run

# Make predictions

train_predictions = forest.predict(train_bag)
valid_predictions = forest.predict(test_bag)

##function
from sklearn.ensemble import RandomForestClassifier

# put everything together in a function

def predict_sentiment(cleaned_reviews, y=train["sentiment"]):

    print("Creating the bag of words model!\n")
    # CountVectorizer" is scikit-learn's bag of words tool, here we show more keywords 
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 2000) 
    
    X_train, X_test, y_train, y_test = train_test_split(\
    cleaned_reviews, y, random_state=0, test_size=.2)

    # Then we use fit_transform() to fit the model / learn the vocabulary,
    # then transform the data into feature vectors.
    # The input should be a list of strings. .toarraty() converts to a numpy array
    
    train_bag = vectorizer.fit_transform(X_train).toarray()
    test_bag = vectorizer.transform(X_test).toarray()

    # You can extract the vocabulary created by CountVectorizer
    # by running print(vectorizer.get_feature_names())


    print("Training the random forest classifier!\n")
    # Initialize a Random Forest classifier with 50 trees
    forest = RandomForestClassifier(n_estimators = 50) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the target variable
    forest = forest.fit(train_bag, y_train)


    train_predictions = forest.predict(train_bag)
    test_predictions = forest.predict(test_bag)
    
    train_acc = metrics.accuracy_score(y_train, train_predictions)
    valid_acc = metrics.accuracy_score(y_test, test_predictions)
    print("The training accuracy is: ", train_acc, "\n", "The validation accuracy is: ", valid_acc)
    return(forest,vectorizer)


print('Original Reviews')
forest1,vec1 = predict_sentiment(review_clean_original)

# Saving model to disk
pickle.dump(forest, open('model.pkl','wb'))
