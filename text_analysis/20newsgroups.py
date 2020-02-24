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


def histogram():
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words='english', max_features=500)
    transformed = cv.fit_transform(groups.data)
    
    sas.distplot(np.log(transformed.toarray().sum(axis=0)))
    plt.xlabel('Log Count')
    plt.ylabel('Frequency')
    plt.title('Distribution Plot of 500 Words Counts')
    plt.show()

def main():
    np.unique(groups.target)
    data_listing()
    data_visualization()
    histogram()
    
if __name__=='__main__':
    groups = fetch_20newsgroups()
    main()
