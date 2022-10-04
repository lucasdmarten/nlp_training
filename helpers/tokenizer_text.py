import nltk
from nltk import tokenize

from wordcloud import WordCloud
import pandas as pd
from unidecode import unidecode
import seaborn as sns
import matplotlib.pyplot as plt
import re
from unidecode import unidecode


def normalize_sentence(sentence):
    norm_sentence = sentence.lower() 
    norm_sentence = re.sub(r'[^\w\s]','', norm_sentence) 
    norm_sentence = norm_sentence.strip() 
    norm_sentence = unidecode(norm_sentence)
    norm_sentence = ' '.join(norm_sentence.split()) 
    return norm_sentence

df = pd.read_csv('example_vcan.csv')
df = df.iloc[:,[1,2,3,4]]

# just a simple classification
for i in range(len(df)):
    df['vcan'].iloc[i] = 1 if "VCAN" in df['text_data'].iloc[i] else 0

white_space_tokenize = tokenize.WhitespaceTokenizer()    
    
stop_words = nltk.corpus.stopwords.words("portuguese")
stop_words.append('o')
stop_words.append('a')
stop_words.append('imagem')
stop_words.append('a')


phrase_pos = list()
for text in df.text_data:
    new_phrase = list()
    phrase_text = white_space_tokenize.tokenize(text)
    for word in phrase_text:
        if word.lower() not in stop_words:
            new_phrase.append(word.lower())
            
    phrase_pos.append(' '.join(new_phrase))

df['treatment_1'] = phrase_pos

with_vcan = df.query("vcan == 1")
with_vcan_words = ' '.join([unidecode(text[30:]) for text in with_vcan.treatment_1]) 

without_vcan = df.query("vcan == 0")
without_vcan_words = ' '.join([unidecode(text[30:]) for text in without_vcan.treatment_1]) 

all_of_words = ' '.join([unidecode(text[30:]) for text in df.treatment_1]) 
without_stop_words = normalize_sentence(' '.join([text for text in df['treatment_1']]))

tokens = white_space_tokenize.tokenize(without_vcan_words)

frequency = nltk.FreqDist(tokens)
df_frequency = pd.DataFrame({"Word":list(frequency.keys()), "Frequency": list(frequency.values())})

def plot_word_clouds(words):
    cloudWords = WordCloud(
        width=800, height=800, max_font_size=110, collocations=False
    )
    fig_cloud = cloudWords.generate(words)
    plt.figure(figsize=(10,20))
    plt.axis('off')
    plt.imshow(fig_cloud, interpolation='bilinear')    
    plt.show()
    
def plot_parret():
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
 
plot_word_clouds(without_vcan_words)