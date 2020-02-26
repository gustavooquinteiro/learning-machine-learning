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

def clean_words(cv):
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
    print(cv.get_feature_names())
    return transformed

def clustering(transformed):
    from sklearn.cluster import KMeans
    
    km = KMeans(n_clusters=20)
    km.fit(transformed)
    labels = groups.target
    plt.scatter(labels, km.labels_)
    plt.xlabel('Newsgroup')
    plt.ylabel('Cluster')
    plt.show()
    

def main(np_unique):
    if np_unique:
        np.unique(groups.target)
    data_listing()
    data_visualization()
    clustering(clean_words(histogram(True)))
    
if __name__=='__main__':
    import sys
    
    groups = fetch_20newsgroups()
    
    np_unique = False if len(sys.argv) > 2 else True
    
    main(np_unique)
