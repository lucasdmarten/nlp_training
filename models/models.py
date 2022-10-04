from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from nltk import tokenize
import re
from unidecode import unidecode
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


STOP_WORDS = nltk.corpus.stopwords.words("portuguese")
MAX_FEATURES = [50, None, 500, 10000]
SOLVERS = ['lbfgs','liblinear']

def load_df():
    df = pd.read_csv('example_words.csv')
    df = df.iloc[:,[1,2,3,4]]
    return df

def add_stop_words(*args):
    for arg in args:
        STOP_WORDS.append(arg)
    return STOP_WORDS

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

def querys(df, conditional_str):
    new_df = df.query(conditional_str)
    join_words = ' '.join([text for text in new_df.treatment_1]) 
    return join_words

def make_frequency_df(token):
    frequency = nltk.FreqDist(token)
    df_frequency = pd.DataFrame({
        "Word": list(frequency.keys()), 
        "Frequency": list(frequency.values())
    })
    return df_frequency

def run_model(df, xcol='treatment_1', ycol='vcan', solver='lbfgs', lowercase=False,
              max_features=50, random_state=42):
    vectorizer = CountVectorizer(lowercase=lowercase, max_features=max_features) 
                                   # lowercase = False (not transform to lowercase)
                                   # max_features (create a vector limited a n features,
                                   #                 of the often persist words)
    bag_of_words = vectorizer.fit_transform(df['treatment_1']) 
                                   # return sparse matrix (a lot of zero values)
    train, test, class_train, class_test =\
        train_test_split(bag_of_words, df['vcan'], random_state=random_state)

    logistic_regression = LogisticRegression(solver=solver) 
    logistic_regression.fit(train, class_train)
    
    accuracy = logistic_regression.score(test, class_test)
    print(accuracy)
    
def plot_word_clouds(words):
    cloudWords = WordCloud(
        width=800, height=800, max_font_size=110, collocations=False
    )
    fig_cloud = cloudWords.generate(words)
    plt.figure(figsize=(10,20))
    plt.axis('off')
    plt.imshow(fig_cloud, interpolation='bilinear')    
    plt.show()

def plot_parret(df_frequency):
    # sugested separe of data -> e de o do a da um em no que com O imagem esta na A 
    data = df_frequency.nlargest(columns='Frequency',n=20)
    total = data['Frequency'].sum()
    data['Percentage'] = data['Frequency'].cumsum() / total * 100

    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=data, x='Word', y='Frequency', color='gray')

    ax2 = ax.twinx()
    sns.lineplot(data=data, x='Word', y='Percentage', color='red', sort=False, ax=ax2)

    ax.set(ylabel='Count')
    ax.set_xticklabels(data.Word, rotation = 25)
    plt.show()
    
if __name__=='__main__':
    df = pd.read_csv('data/example_words.csv')
    white_space_tokenize = tokenize.WhitespaceTokenizer()  
    add_stop_words(
        '250', 'altitude', 'hpa', 'jato', 'influencia', 'atua', 'ramo', 'parte',
        'notase','norte','sul', 'area', 'observase', 'regiao', 'imagem', 'sobre',
        'e', 'de', 'o', 'do', 'a', 'da', 'um', 'em', 'no', 'que', 'com', 'O', 'esta',
        'na', 'A', 
        
    )
    df = process_with_stop_words(df, 'text_data', 'treatment_1', white_space_tokenize)

    join_texts_with_vcan = querys(df, 'difluência == 1')
    join_texts_without_vcan = querys(df, 'difluência == 0')
    tokens_with_vcan = white_space_tokenize.tokenize(join_texts_with_vcan)
    tokens_without_vcan = white_space_tokenize.tokenize(join_texts_without_vcan)
    df_frequency_with_vcan = make_frequency_df(tokens_with_vcan)
    df_frequency_without_vcan = make_frequency_df(tokens_without_vcan)
    
    plot_word_clouds(join_texts_without_vcan)
    plot_parret(df_frequency_without_vcan)
    
    run_model(df, xcol='treatment_1', ycol='difluência', 
              solver=SOLVERS[0], lowercase=True, max_features=10,  random_state=112222)