from json import load

def read_config(path):
    cfg = open(path, 'r')
    return load(cfg)