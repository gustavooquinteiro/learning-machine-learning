from sklearn.datasets import fetch_20newsgroups
import numpy as np

def main():
    groups = fetch_20newsgroups()
    np.unique(groups.target)
    for i in range(len(groups.target)):
        print("Target {} is {}" .format(groups.target[i], groups.target_names[groups.target[i]]))
        print("Data: ")
        print(groups.data[i])
    
if __name__=='__main__':
    main()
