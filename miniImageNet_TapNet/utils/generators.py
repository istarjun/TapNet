"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot
"""
import numpy as np
import random

class miniImageNetGenerator(object):
    """miniImageNetGenerator

    Args:
        data_file (str): 'data/train.npz' or 'data/test.npz' 
        nb_classes (int): number of classes in an episode
        nb_samples_per_class (int): nuber of samples per class in an episode
        max_iter (int): max number of episode generation
        xp: numpy or cupy
    """
    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=10, 
                  max_iter=None, xp=np):
        super(miniImageNetGenerator, self).__init__()
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(self.data_file)

    def _load_data(self, data_file):
        data_dict = np.load(data_file)
        return {key: np.array(val) for (key, val) in data_dict.items()}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class) 
            
            return (self.num_iter - 1), (images, labels) 
        else:
            raise StopIteration()

    def sample(self, nb_classes, nb_samples_per_class):
        sampled_characters = random.sample(self.data.keys(), nb_classes) 
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class) 
            labels_and_images.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind])
        arg_labels_and_images = [] 
        for i in range(self.nb_samples_per_class):   
            for j in range(self.nb_classes):                        
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        return images, labels

