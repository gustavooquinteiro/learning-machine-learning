from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import seaborn as sas
import numpy as np

def data_listing(shows=False):
    if shows:
        for i in range(len(groups.target)):
            print("Target {} is {}" .format(groups.target[i], groups.target_names[groups.target[i]]))
            print("Data: ")
            print(groups.data[i])

def data_visualization(shows=False):
    if shows:
        sas.distplot(groups.target)
        plt.show()


def histogram(skip_plot=False):
    from sklearn.feature_extraction.text import CountVectorizer
    
    cv = CountVectorizer(stop_words='english', max_features=500)
    transformed = cv.fit_transform(groups.data)
    if not skip_plot:
        sas.distplot(np.log(transformed.toarray().sum(axis=0)))
        plt.xlabel('Log Count')
        plt.ylabel('Frequency')
        plt.title('Distribution Plot of 500 Words Counts')
        plt.show()
    else:
        return cv

def clean_words(cv, prints=False):
    from nltk.corpus import names
    from nltk.stem import WordNetLemmatizer
    
    cleaned = []
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    
    for post in groups.data:
        cleaned.append(' '.join([lemmatizer.lemmatize(word.lower())
                                for word in post.split()
                                if word.isalpha() 
                                and word not in all_names]))
        
    transformed = cv.fit_transform(cleaned)
    if prints:
        print(cv.get_feature_names())
    return transformed

def clustering(transformed, show=True):
    if show:
        from sklearn.cluster import KMeans
        
        km = KMeans(n_clusters=20)
        km.fit(transformed)
        labels = groups.target
        plt.scatter(labels, km.labels_)
        plt.xlabel('Newsgroup')
        plt.ylabel('Cluster')
        plt.show()
    
def topic_modeling(transformed, cv):
    from sklearn.decomposition import NMF
    
    nmf = NMF(n_components=100, random_state=43).fit(transformed)
    for topic_idx, topic in enumerate(nmf.components_):
        label = '{}: '.format(topic_idx)
        print(label, ' '.join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))
    

def main(np_unique):
    if np_unique:
        np.unique(groups.target)
    data_listing()
    data_visualization()
    
    cv = histogram(True)
    
    transformed = clean_words(cv)
    
    clustering(transformed, False)
    
    topic_modeling(transformed, cv)
    
if __name__=='__main__':
    import sys
    
    groups = fetch_20newsgroups()
    
    np_unique = False if len(sys.argv) > 2 else True
    
    main(np_unique)
