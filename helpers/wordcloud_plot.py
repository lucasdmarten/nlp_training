from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from unidecode import unidecode

df = pd.read_csv('example_vcan.csv')
df = df.iloc[:,[1,2,3,4]]

# just a simple classification
for i in range(len(df)):
    df['vcan'].iloc[i] = 1 if "VCAN" in df['text_data'].iloc[i] else 0

def plot_word_clouds(words):
    cloudWords = WordCloud(
        width=800, height=800, max_font_size=110, collocations=False
    )
    fig_cloud = cloudWords.generate(words)
    plt.figure(figsize=(10,20))
    plt.axis('off')
    plt.imshow(fig_cloud, interpolation='bilinear')    
    plt.savefig(f'./{words}.png')
    plt.close()
with_vcan = df.query("vcan == 1")
with_vcan_words = ' '.join([unidecode(text[30:]) for text in with_vcan.text_data]) 

without_vcan = df.query("vcan == 0")
without_vcan_words = ' '.join([unidecode(text[30:]) for text in without_vcan.text_data]) 
all_of_words = ' '.join([unidecode(text[30:]) for text in df.text_data]) 

for words in [with_vcan_words, without_vcan_words, all_of_words]:
    plot_word_clouds(words)