import nltk
from nltk import tokenize

import pandas as pd
from unidecode import unidecode
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('example_vcan.csv')
df = df.iloc[:,[1,2,3,4]]

# just a simple classification
for i in range(len(df)):
    df['vcan'].iloc[i] = 1 if "VCAN" in df['text_data'].iloc[i] else 0

with_vcan = df.query("vcan == 1")
with_vcan_words = ' '.join([unidecode(text[30:]) for text in with_vcan.text_data]) 

without_vcan = df.query("vcan == 0")
without_vcan_words = ' '.join([unidecode(text[30:]) for text in without_vcan.text_data]) 
all_of_words = ' '.join([unidecode(text[30:]) for text in df.text_data]) 

white_space_tokenize = tokenize.WhitespaceTokenizer()
tokens = white_space_tokenize.tokenize(with_vcan_words)

frequency = nltk.FreqDist(tokens)
df_frequency = pd.DataFrame({"Word":list(frequency.keys()), "Frequency": list(frequency.values())})

# sugested separe of data -> e de o do a da um em no que com O imagem esta na A 
data = df_frequency.nlargest(columns='Frequency',n=10)
total = data['Frequency'].sum()
data['Percentage'] = data['Frequency'].cumsum() / total * 100

plt.figure(figsize=(12,8))
ax = sns.barplot(data=data, x='Word', y='Frequency', color='gray')

ax2 = ax.twinx()
sns.lineplot(data=data, x='Word', y='Percentage', color='red', sort=False, ax=ax2)

ax.set(ylabel='Count')
plt.show()