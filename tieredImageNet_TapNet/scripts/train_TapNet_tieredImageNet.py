import os
import sys
sys.path.append('../')
import argparse

import numpy as np
import scipy.io as sio

import chainer.functions as F
from chainer import optimizers
from chainer import cuda
from chainer import serializers

from utils.generators import tieredImageNetGenerator
from utils.model_TapNet_ResNet12 import TapNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu device number. -1 for cpu.')   
    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of shots.')  
    parser.add_argument('--nb_class_train', type=int, default=20,
                        help='Number of training classes .')      
    parser.add_argument('--nb_class_test', type=int, default=5,
                        help='Number of test classes .')    
    parser.add_argument('--n_query_train', type=int, default=8,
                        help='Number of queries per class in training.')    
    parser.add_argument('--n_query_test', type=int, default=15,
                        help='Number of queries per class in test.')    
    parser.add_argument('--wd_rate', type=float, default=0,
                        help='Weight decay rate in Adam optimizer')  
    # set params
    # -----------
    args = parser.parse_args()        
    if args.gpu < 0:
        xp = np
    else:
        import cupy as cp
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="%d" %args.gpu
        xp = cp
    dimension=512
    max_iter=50001
    lrdecay = True
    lrstep = 40000
    n_shot=args.n_shot
    n_query=args.n_query_train
    n_query_test = args.n_query_test
    nb_class_train=args.nb_class_train
    nb_class_test=args.nb_class_test
    wd_rate=args.wd_rate
    savefile_name='save/TapNet_tieredImageNet_ResNet12.mat'
    filename_5shot='save/TapNet_tieredImageNet_ResNet12'
    filename_5shot_last='save/TapNet_tieredImageNet_ResNet12_last'

    # set up training
    # ------------------
    model = TapNet(nb_class_train=nb_class_train, nb_class_test=nb_class_test, input_size=3*84*84, 
                dimension=dimension, n_shot=n_shot, gpu=args.gpu)
    
    optimizer = optimizers.Adam(alpha=1e-3, weight_decay_rate=wd_rate)
    model.set_optimizer(optimizer)

    train_generator = tieredImageNetGenerator(image_file='../data/train_images.npz', label_file='../data/train_labels.pkl', 
                                        nb_classes=nb_class_train, nb_samples_per_class=n_shot+n_query, 
                                        max_iter=max_iter, xp=xp)

    # Result analysis list 
    # -----------------
    
    loss_h=[]
    accuracy_h_val=[]
    accuracy_h_test=[]

    acc_best=0
    epoch_best=0

    # start training
    # ----------------

    for t, (images, labels) in train_generator:
        # train
        loss = model.train(images, labels)
        # logging 
        loss_h.extend([loss.tolist()])
        if (t % 50 == 0):
            print("Episode: %d, Train Loss: %f "%(t, loss))
    
        if (t != 0) and (t % 500 == 0):                
            print('Evaluation in Validation data')
            test_generator = tieredImageNetGenerator(image_file='../data/val_images.npz', label_file='../data/val_labels.pkl', 
                                               nb_classes=nb_class_test, nb_samples_per_class=n_shot+n_query_test, 
                                               max_iter=600, xp=xp)
            scores = []                                              
            for i, (images, labels) in test_generator:
                accs = model.evaluate(images, labels)                
                accs_ = [cuda.to_cpu(acc) for acc in accs]
                score = np.asarray(accs_, dtype=int)
                scores.append(score)
            print(('Accuracy 5 shot ={:.2f}%').format(100*np.mean(np.array(scores))))
            accuracy_t=100*np.mean(np.array(scores))
            
            if acc_best < accuracy_t:
                acc_best = accuracy_t
                epoch_best=t
                serializers.save_npz(filename_5shot,model.chain)
                
            accuracy_h_val.extend([accuracy_t.tolist()])
            del(test_generator)
            del(accs)
            del(accs_)
            del(accuracy_t)
            
            print('Evaluation in Test data')
            test_generator = tieredImageNetGenerator(image_file='../data/test_images.npz', label_file='../data/test_labels.pkl', 
                                               nb_classes=nb_class_test, nb_samples_per_class=n_shot+n_query_test, 
                                               max_iter=600, xp=xp)
            scores = []                                              
            for i, (images, labels) in test_generator:
                accs = model.evaluate(images, labels)                
                accs_ = [cuda.to_cpu(acc) for acc in accs]
                score = np.asarray(accs_, dtype=int)
                scores.append(score)
            print(('Accuracy 5 shot ={:.2f}%').format(100*np.mean(np.array(scores))))
            accuracy_t=100*np.mean(np.array(scores))
            accuracy_h_test.extend([accuracy_t.tolist()])
            del(test_generator)
            del(accs)
            del(accs_)
            del(accuracy_t)
            sio.savemat(savefile_name, {'accuracy_h_val':accuracy_h_val, 'accuracy_h_test':accuracy_h_test, 'epoch_best':epoch_best,'acc_best':acc_best})
            if len(accuracy_h_val) >10:
                print('***Average accuracy on past 10 evaluation***')             
                print('Best epoch =',epoch_best,'Best 5 shot acc=',acc_best)
                
            serializers.save_npz(filename_5shot_last,model.chain)
    
        if (t != 0) and (t % lrstep == 0) and lrdecay:
            model.decay_learning_rate(0.1)

    
    accuracy_h5=[]

    serializers.load_npz(filename_5shot, model.chain)
    print('Evaluating the best 5shot model...') 
    for i in range(50):
        test_generator = tieredImageNetGenerator(image_file='../data/test_images.npz', label_file='../data/test_labels.pkl', 
                                               nb_classes=nb_class_test, nb_samples_per_class=n_shot+n_query_test, 
                                               max_iter=600, xp=xp)
        scores=[]
        for j, (images, labels) in test_generator:
            accs = model.evaluate(images, labels)                
            accs_ = [cuda.to_cpu(acc) for acc in accs]
            score = np.asarray(accs_, dtype=int)
            scores.append(score)
        accuracy_t=100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
        print(('600 episodes with 15-query accuracy: 5-shot ={:.2f}%').format(accuracy_t))
        del(test_generator)
        del(accs)
        del(accs_)
        del(accuracy_t)   
        sio.savemat(savefile_name, {'accuracy_h_val':accuracy_h_val, 'accuracy_h_test':accuracy_h_test, 'epoch_best':epoch_best,'acc_best':acc_best, 'accuracy_h5':accuracy_h5})
    print(('Accuracy_test 5 shot ={:.2f}%').format(np.mean(accuracy_h5)))


