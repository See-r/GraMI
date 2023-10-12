import numpy as np
import scipy
import scipy.sparse as sp
import pickle
import torch

np.set_printoptions(threshold=np.inf)

def normalized(input_data):
    input_data = np.array(input_data, dtype=float)

    length = input_data.shape[0]
    for i in range(0, length):
        MIN, MAX = np.min(input_data[i]), np.max(input_data[i])
        if MIN == 0 and MAX == 0:
            continue
        elif MAX > MIN:
            input_data[i] = (input_data[i] - MIN) / (MAX - MIN)
        elif MAX == MIN:
            j = np.nonzero(input_data[i])
            input_data[i][j] = 1.
    return input_data


def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    # 0 for papers, 1 for authors, 2 for subjects
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]  # list[str]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    features_0 = normalized(features_0)
    features_1 = normalized(features_1)
    features_2 = normalized(features_2)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    type_mask = np.load(prefix + '/node_types.npy')

    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [adjlist00, adjlist01], \
        [idx00, idx01], \
        [features_0, features_1, features_2], \
        adjM, \
        type_mask, \
        labels,\
        train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    # 0 for author, 1 for paper, 2 for term , 3 for venue
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    # 0 for authors, 1 for papers, 2 for terms, 3 for venue
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    features_0 = normalized(features_0)
    features_1 = normalized(features_1)
    features_2 = normalized(features_2)
    features_3 = normalized(features_3)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
        [idx00, idx01, idx02], \
        [features_0, features_1, features_2, features_3], \
        adjM, \
        type_mask, \
        labels,\
        train_val_test_idx

def load_YELP_data(prefix='data/preprocessed/YELP_processed'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    features_3 = scipy.sparse.load_npz(prefix + '/features_3.npz').toarray()

    features_0 = normalized(features_0)
    features_1 = normalized(features_1)
    features_2 = normalized(features_2)
    features_3 = normalized(features_3)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [features_0, features_1, features_2, features_3], \
        adjM, \
        type_mask, \
        labels, \
        train_val_test_idx
