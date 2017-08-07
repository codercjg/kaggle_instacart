# -- coding: utf-8 --
import pickle

def model_save(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)
    
def model_load(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj