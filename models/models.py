import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from nltk import tokenize
import re
from unidecode import unidecode


STOP_WORDS = nltk.corpus.stopwords.words("portuguese")
MAX_FEATURES = [50, None, 500, 10000]
SOLVERS = ['lbfgs','liblinear']

def load_df():
    df = pd.read_csv('example_vcan.csv')
    df = df.iloc[:,[1,2,3,4]]
    return df

def normalize_sentence(sentence):
    norm_sentence = sentence.lower() 
    norm_sentence = re.sub(r'[^\w\s]','', norm_sentence) 
    norm_sentence = norm_sentence.strip() 
    norm_sentence = unidecode(norm_sentence)
    norm_sentence = ' '.join(norm_sentence.split()) 
    return norm_sentence

def is_vcan(df):
    for i in range(len(df)):
        df['vcan'].iloc[i] = 1 if "VCAN" in df['text_data'].iloc[i] else 0
    return df

def process_with_stop_words(df, col, new_col, white_space_tokenize):
    phrase_pos = list()
    for text in df[col]:
        new_phrase = list()
        phrase_text = white_space_tokenize.tokenize(
            normalize_sentence(text)
        )
        for word in phrase_text:
            if word not in STOP_WORDS:
                new_phrase.append(word.lower())

        phrase_pos.append(' '.join(new_phrase))
    df[new_col] = phrase_pos
    return df

def run_model(solver='lbfgs', lowercase=False,
              max_features=50, random_state=42):
    df = pd.read_csv('example_vcan.csv')
    df = is_vcan(df)
    white_space_tokenize = tokenize.WhitespaceTokenizer()    
    df = process_with_stop_words(df, 'text_data', 'treatment_1', white_space_tokenize)
    vectorizer = CountVectorizer(lowercase=lowercase, max_features=max_features) 
                                   # lowercase = False (not transform to lowercase)
                                   # max_features (create a vector limited a n features,
                                   #                 of the often persist words)
    bag_of_words = vectorizer.fit_transform(df.treatment_1) 
                                   # return sparse matrix (a lot of zero values)

    train, test, class_train, class_test = train_test_split(bag_of_words,
                                                                  df.vcan,
                                                                  random_state=random_state)

    logistic_regression = LogisticRegression(solver=solver) 
                                    # solver
                                    # max_iter=1000 be carefoul, consum of RAM
    logistic_regression.fit(train, class_train)
    accuracy = logistic_regression.score(test, class_test)
    print(accuracy)
    
    
if __name__=='__main__':
    for solver in SOLVERS:
        for max_feature in MAX_FEATURES:
            run_model(solver=solver, lowercase=False, max_features=max_feature,
                        random_state=42)