# TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning

Code for the ICML 2019 paper TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning

## Dependencies
* This code is tested on Ubuntu 16.04 with Python 3.6 and chainer 5.20

## Data  
### miniImageNet
#Download and unzip "mini-imagenet.tar.gz" from Google Drive link [[mini-ImageNet](https://drive.google.com/file/d/1DvYd7LMa0zvlqTM8oBdCWwQSxpZdf_D5/view?usp=sharing)]

#Place ``train.npz``, ``val.npz``, ``test.npz`` files in ``TapNet/miniImageNet_TapNet/data``


### tieredImageNet
#Download and unzip "tiered-imagenet.tar.gz" from Google Drive link [[tiered-ImageNet](https://drive.google.com/file/d/1zz7bAYus7EeoMokwUQlLc3OY_eoII8B7/view?usp=sharing)]

#Place images ``.npz`` and labels ``.pkl`` files in ``TapNet/tieredImageNet_TapNet/data``

## Running the code

```
#For miniImageNet experiment

cd /TapNet/miniImageNet_TapNet/scripts
python train_TapNet_miniImageNet.py --gpu {GPU device number}
                                    --n_shot {n_shot}
                                    --nb_class_train {number of classes in training}
                                    --nb_class_test {number of classes in test}
                                    --n_query_train {number of queries per class in training}
                                    --n_query_test {number of queries per class in test}
                                    --wd_rate {Weight decay rate}
                                    
                                    
#For tieredImageNet experiment

cd /TapNet/tieredImageNet_TapNet/scripts
python train_TapNet_tieredImageNet.py --gpu {GPU device number}
                                    --n_shot {n_shot}
                                    --nb_class_train {number of classes in training}
                                    --nb_class_test {number of classes in test}
                                    --n_query_train {number of queries per class in training}
                                    --n_query_test {number of queries per class in test}
                                    --wd_rate {Weight decay rate}
```

