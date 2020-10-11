import pickle
import hydra
from omegaconf import DictConfig
import PIL.Image as Image



def save_pkl(ph, pkl):
    with open(ph, 'wb') as f:
        pickle.dump(pkl, f, 0)


def load_pkl(ph):
    with open(ph, 'rb') as f:
        pkl = pickle.load(f)
    return pkl


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
