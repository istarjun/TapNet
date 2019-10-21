"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot
"""
import numpy as np
import random
import pickle as pkl

class tieredImageNetGenerator(object):
    """tieredImageNetGenerator

    Args:
        image_file (str): 'data/train_images.npz' or 'data/test_images.npz' or 'data/val_images.npz' 
        label_file (str): 'data/train_labels.pkl' or 'data/test_labels.pkl' or 'data/val_labels.pkl'
        nb_classes (int): number of classes in an episode
        nb_samples_per_class (int): nuber of samples per class in an episode
        max_iter (int): max number of episode generation
        xp: numpy or cupy
    """
    def __init__(self, image_file, label_file, nb_classes=5, nb_samples_per_class=10, 
                  max_iter=None, xp=np):
        super(tieredImageNetGenerator, self).__init__()
        self.image_file = image_file
        self.label_file = label_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self._load_data(self.image_file, self.label_file)

    def _load_data(self, image_file, label_file):
        with np.load(image_file, mmap_mode="r", encoding='latin1') as data:
            images=data["images"]
        with open(label_file,'rb') as f:
            data=pkl.load(f, encoding='bytes')
            label_specific = data[b"label_specific"]
            label_specific_str = data[b"label_specific_str"]
        num_ex = label_specific.shape[0]
        ex_ids = np.arange(num_ex)
        num_label_cls_specific = len(label_specific_str)
        self.label_specific_idict={}
        for cc in range(num_label_cls_specific):
            self.label_specific_idict[cc]=ex_ids[label_specific == cc]
        self.images=images

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
        sampled_characters = random.sample(self.label_specific_idict.keys(), nb_classes) 
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            _ind = random.sample(list(self.label_specific_idict[char]), nb_samples_per_class) 
            labels_and_images.extend([(k, self.xp.array(self.images[i]/np.float32(255).flatten())) for i in _ind])
        arg_labels_and_images = [] 
        for i in range(self.nb_samples_per_class):     
            for j in range(self.nb_classes):                        
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        return images, labels
