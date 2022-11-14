import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

from lib.dataset import clean_and_truncate
from lib.embedder import Embedder

# load and clean dataset
def load_dataset(datadir):
    dataset = pd.read_csv(os.path.join(datadir, 'testset_C.csv'), sep=';')

    # create labels
    encoder = LabelEncoder()
    encoder.fit(dataset['productgroup'])
    dataset['class'] = encoder.transform(dataset['productgroup'])

    # prepare and clean dataset
    dataset['main_text_cl'] = dataset['main_text'].apply(lambda x: clean_and_truncate(x, 20))
    dataset['add_text_cl'] = dataset['add_text'].apply(lambda x: clean_and_truncate(x, 2))
    dataset['manufacturer_cl'] = dataset['manufacturer'].apply(lambda x: clean_and_truncate(x, 1))

    # generate the input text required
    dataset['input_text'] = dataset.manufacturer_cl + ' ' + dataset.add_text_cl + ' ' + dataset.main_text_cl
    dataset['input_text'] = dataset.input_text.str.strip()
    dataset['input_text'] = dataset.input_text.replace('', np.nan)

    # remove NAs in input text
    dataset = dataset[~dataset.input_text.isna()]
    return dataset, encoder


def get_embeddings(dataset):
    """ generate embeddings for the dataset """
    embedder = Embedder(max_len=25)

    all_texts = dataset.input_text.to_list()

    # process in batches to better handle memory
    # TODO: make this better memory managed!
    def _batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    print('generate embeddings (in batches)....')
    outputs = []
    for i, texts in enumerate(_batch(all_texts, 100)):
        print(f'{i} / {len(all_texts) / 100}')
        embeddings = embedder.transform(texts)
        outputs.append(embeddings)

    X = np.concatenate(outputs, axis=0)
    y = dataset['class'].to_numpy()

    return X, y


def main():
    import argparse
    parser = argparse.ArgumentParser('Generate embeddings for the given dataset')
    parser.add_argument('--datadir', type=str, help='path to load dataset from', default='../data/')
    parser.add_argument('--savedir', type=str, help='path to store the product embeddings', default='../results/')
    args = parser.parse_args()

    dataset, encoder = load_dataset(args.datadir)
    X, y = get_embeddings(dataset)

    # store label encoder and embeddings
    np.save(os.path.join(args.datadir, 'X.npy'), X)
    np.save(os.path.join(args.datadir, 'y.npy'), y)
    os.makedirs(args.savedir, exist_ok=True)
    with open(os.path.join(args.savedir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)


if __name__ == '__main__':
    main()
