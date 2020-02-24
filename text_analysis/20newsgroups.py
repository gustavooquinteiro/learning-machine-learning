from sklearn.datasets import fetch_20newsgroups

def main():
    groups=fetch_20newsgroups()
    print(groups)
    
if __name__=='__main__':
    main()
