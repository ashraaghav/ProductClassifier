import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import os


def train_model(datadir, savedir):
    # load pre-computed embeddings from the dataset for reference
    X = np.load(os.path.join(datadir, 'X.npy'))
    y = np.load(os.path.join(datadir, 'y.npy'))

    # flatten the input for SVM
    X = X.reshape(X.shape[0], -1)

    # split into train, validation & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=1, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8,
                                                          random_state=1, stratify=y_train)

    svc = SVC(kernel='rbf', C=10.0, random_state=1)

    # fit model
    svc.fit(X_train, y_train)

    # store model
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, 'category_classifier.pkl'), 'wb') as f:
        pickle.dump(obj=svc, file=f)


def main():
    import argparse
    parser = argparse.ArgumentParser('Train model for the embeddings')
    parser.add_argument('--datadir', type=str, help='path to load dataset from', default='../data/')
    parser.add_argument('--savedir', type=str, help='path to store the product embeddings', default='../results/')
    args = parser.parse_args()

    train_model(args.datadir, args.savedir)


if __name__ == '__main__':
    main()
