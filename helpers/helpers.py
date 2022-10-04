TEXT = ["Assisti um filme ótimo", "Assisti um filme ruim"]


def how_vectorizer_text():
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(lowercase=False, max_features=50) 
                                   # lowercase = False (not transform to lowercase)
                                   # max_features (create a vector limited a n features,
                                   #                of the often persist words)
    bag_of_words = vectorizer.fit_transform(TEXT) 
                                   # return sparse matrix (a lot of zero values)
    
def get_feature_names(vectorizer):
    # ['assisti', 'filme', 'ruim', 'um', 'ótimo']
    return vectorizer.get_feature_names()

def parse_dataframe(bag_of_words):
    import pandas as pd
    sparse_matrix = pd.SparseDataFrame(bag_of_words, 
                      columns=vectorizer.get_feature_nams())
    return sparse_matrix

def cloud_words():
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    all_text_joined = ' '.join(TEXT)
    cloudWords = WordCloud().generate(all_text_joined)
    plt.figure(); plt.imshow(cloudWords)