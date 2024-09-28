import pickle
import os


def save_pickle(obj, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
