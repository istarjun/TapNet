import cupy as cp
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda

from utils.rank_nullspace import nullspace_gpu

class TapNet(object):
    def __init__(self, nb_class_train, nb_class_test,  input_size, dimension, 
                 n_shot, gpu=-1):
        """
        Args
            nb_class_train (int): number of classes in a training episode
            nb_class_test (int): number of classes in a test episode
            input_size (int): dimension of input vector
            dimension (int) : dimension of embedding space
            n_shot (int) : number of shots
        """
        self.nb_class_train = nb_class_train
        self.nb_class_test = nb_class_test
        self.input_size = input_size
        self.dimension = dimension 
        self.n_shot = n_shot
        # create chain
        self.chain = self._create_chain()
        self.set_gpu(gpu)


    # Set up methods
    # ---------------
    @property
    def xp(self):
        if self.gpu<0:
            return np
        else:
            return cp

    def set_gpu(self, gpu):
        self.gpu = gpu
        if self.gpu < 0:
            self.chain.to_cpu()
        else:
            self.chain.to_gpu()


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.setup(self.chain)
        self.optimizer.use_cleargrads(use=False)



    def _create_chain(self):
        chain = chainer.Chain(
            l_conv1_1=L.Convolution2D(None,64,(3,3), pad=1),
            l_norm1_1=L.BatchNormalization(64),
            l_conv1_2=L.Convolution2D(64,64,(3,3), pad=1),
            l_norm1_2=L.BatchNormalization(64),
            l_conv1_3=L.Convolution2D(64,64,(3,3), pad=1),
            l_norm1_3=L.BatchNormalization(64),
            l_conv1_r=L.Convolution2D(None,64,(3,3), pad=1),
            l_norm1_r=L.BatchNormalization(64),
            
            l_conv2_1=L.Convolution2D(64,128,(3,3), pad=1),
            l_norm2_1=L.BatchNormalization(128),
            l_conv2_2=L.Convolution2D(128,128,(3,3), pad=1),
            l_norm2_2=L.BatchNormalization(128),
            l_conv2_3=L.Convolution2D(128,128,(3,3), pad=1),
            l_norm2_3=L.BatchNormalization(128),  
            l_conv2_r=L.Convolution2D(64,128,(3,3), pad=1),
            l_norm2_r=L.BatchNormalization(128),

            l_conv3_1=L.Convolution2D(128,256,(3,3), pad=1),
            l_norm3_1=L.BatchNormalization(256),
            l_conv3_2=L.Convolution2D(256,256,(3,3), pad=1),
            l_norm3_2=L.BatchNormalization(256),
            l_conv3_3=L.Convolution2D(256,256,(3,3), pad=1),
            l_norm3_3=L.BatchNormalization(256),
            l_conv3_r=L.Convolution2D(128,256,(3,3), pad=1),
            l_norm3_r=L.BatchNormalization(256),
            
            l_conv4_1=L.Convolution2D(256,512,(3,3), pad=1),
            l_norm4_1=L.BatchNormalization(512),
            l_conv4_2=L.Convolution2D(512,512,(3,3), pad=1),
            l_norm4_2=L.BatchNormalization(512),
            l_conv4_3=L.Convolution2D(512,512,(3,3), pad=1),
            l_norm4_3=L.BatchNormalization(512),
            l_conv4_r=L.Convolution2D(256,512,(3,3), pad=1),
            l_norm4_r=L.BatchNormalization(512),
            
            l_phi=L.Linear(self.dimension, self.nb_class_train),
            )
        return chain


    # Train methods
    # ---------------

    def encoder(self, x, batchsize, train=True):
        with chainer.using_config('train', train):
            x2 = F.reshape(x, (batchsize,84,84,3))
            x3 = F.transpose(x2, [0,3,1,2])
            
            c1_r=self.chain.l_conv1_r(x3)
            n1_r=self.chain.l_norm1_r(c1_r)
            
            c1_1=self.chain.l_conv1_1(x3)
            n1_1=self.chain.l_norm1_1(c1_1)
            a1_1=F.relu(n1_1)
            
            c1_2=self.chain.l_conv1_2(a1_1)
            n1_2=self.chain.l_norm1_2(c1_2)
            a1_2=F.relu(n1_2)
            
            c1_3=self.chain.l_conv1_3(a1_2)
            n1_3=self.chain.l_norm1_3(c1_3)
            
            a1_3=F.relu(n1_3+n1_r)
            
            p1=F.max_pooling_2d(a1_3,2)
            p1=F.dropout(p1,ratio=0.2)
            
            c2_r=self.chain.l_conv2_r(p1)
            n2_r=self.chain.l_norm2_r(c2_r)
            
            c2_1=self.chain.l_conv2_1(p1)
            n2_1=self.chain.l_norm2_1(c2_1)
            a2_1=F.relu(n2_1)
            
            c2_2=self.chain.l_conv2_2(a2_1)
            n2_2=self.chain.l_norm2_2(c2_2)
            a2_2=F.relu(n2_2)
            
            c2_3=self.chain.l_conv2_3(a2_2)
            n2_3=self.chain.l_norm2_3(c2_3)
            
            a2_3=F.relu(n2_3+n2_r)
            
            p2=F.max_pooling_2d(a2_3,2)
            p2=F.dropout(p2, ratio=0.2)
            c3_r=self.chain.l_conv3_r(p2)
            n3_r=self.chain.l_norm3_r(c3_r)
            
            c3_1=self.chain.l_conv3_1(p2)
            n3_1=self.chain.l_norm3_1(c3_1)
            a3_1=F.relu(n3_1)
            
            c3_2=self.chain.l_conv3_2(a3_1)
            n3_2=self.chain.l_norm3_2(c3_2)
            a3_2=F.relu(n3_2)
            
            c3_3=self.chain.l_conv3_3(a3_2)
            n3_3=self.chain.l_norm3_3(c3_3)
            
            a3_3=F.relu(n3_3+n3_r)
            
            p3=F.max_pooling_2d(a3_3,2)
            p3=F.dropout(p3,ratio=0.2)
            
            c4_r=self.chain.l_conv4_r(p3)
            n4_r=self.chain.l_norm4_r(c4_r)
            
            c4_1=self.chain.l_conv4_1(p3)
            n4_1=self.chain.l_norm4_1(c4_1)
            a4_1=F.relu(n4_1)
            
            c4_2=self.chain.l_conv4_2(a4_1)
            n4_2=self.chain.l_norm4_2(c4_2)
            a4_2=F.relu(n4_2)
            
            c4_3=self.chain.l_conv4_3(a4_2)
            n4_3=self.chain.l_norm4_3(c4_3)
            
            a4_3=F.relu(n4_3+n4_r)
            
            p4=F.max_pooling_2d(a4_3,2)
            p4=F.dropout(p4, ratio=0.2)
            
            p5=F.average_pooling_2d(p4,6)
            h_t=F.reshape(p5, (batchsize,-1))
        return  h_t
    
   
    
    def Projection_Space(self, average_key, batchsize, nb_class, train=True, phi_ind=None):
        c_t = average_key
        eps=1e-6
        if train == True:
            Phi_tmp = self.chain.l_phi.W
        else:
            Phi_data = self.chain.l_phi.W.data
            Phi_tmp = chainer.Variable(Phi_data[phi_ind,:])
        for i in range(nb_class):
            if i == 0:
                Phi_sum = Phi_tmp[i]
            else:
                Phi_sum += Phi_tmp[i]
        Phi = nb_class*(Phi_tmp)-F.broadcast_to(Phi_sum,(nb_class,self.dimension))
                    
        power_Phi = F.sqrt(F.sum(Phi*Phi, axis=1))
        power_Phi = F.transpose(F.broadcast_to(power_Phi, [self.dimension,nb_class]))
        
        Phi = Phi/(power_Phi+eps)
        
        power_c = F.sqrt(F.sum(c_t*c_t, axis=1))
        power_c = F.transpose(F.broadcast_to(power_c, [self.dimension,nb_class]))
        c_tmp = c_t/(power_c+eps)
        
        null=Phi - c_tmp
        M = nullspace_gpu(null.data)
        M = F.broadcast_to(M,[batchsize, self.dimension, self.dimension-nb_class])
    
        return M
    
    def compute_power(self, batchsize,key,M, nb_class, train=True,phi_ind=None):
        if train == True:
            Phi_out = self.chain.l_phi.W
        else:
            Phi_data = self.chain.l_phi.W.data
            Phi_out = chainer.Variable(Phi_data[phi_ind,:])
        Phi_out_batch = F.broadcast_to(Phi_out,[batchsize,nb_class, self.dimension])
        PhiM = F.batch_matmul(Phi_out_batch,M)
        PhiMs = F.sum(PhiM*PhiM,axis=2)
        
        key_t = F.reshape(key,[batchsize,1,self.dimension])
        keyM = F.batch_matmul(key_t,M)
        keyMs = F.sum(keyM*keyM, axis=2)
        keyMs = F.broadcast_to(keyMs, [batchsize,nb_class])
        
        pow_t = PhiMs + keyMs
        
        return pow_t
    
    
    def compute_power_avg_phi(self, batchsize, nb_class, average_key, train=False):
        avg_pow = F.sum(average_key*average_key,axis=1)
        Phi = self.chain.l_phi.W
        Phis = F.sum(Phi*Phi,axis=1)
        
        avg_pow_bd = F.broadcast_to(F.reshape(avg_pow,[len(avg_pow),1]),[len(avg_pow),len(Phis)])
        wzs_bd = F.broadcast_to(F.reshape(Phis,[1,len(Phis)]),[len(avg_pow),len(Phis)])
        
        pow_avg = avg_pow_bd + wzs_bd
        
        return pow_avg
    
    
    def compute_loss(self, t_data, r_t, pow_t, batchsize,nb_class, train=True):
        t = chainer.Variable(self.xp.array(t_data, dtype=self.xp.int32)) 
        u = 2*self.chain.l_phi(r_t)-pow_t
        return F.softmax_cross_entropy(u,t)

    def compute_accuracy(self, t_data, r_t, pow_t,batchsize, nb_class, phi_ind=None):
        ro = 2*self.chain.l_phi(r_t)
        ro_t = chainer.Variable(ro.data[:,phi_ind])
        u = ro_t-pow_t
       
        t_est = self.xp.argmax(F.softmax(u).data, axis=1)

        return (t_est == self.xp.array(t_data))
    
    def select_phi(self, average_key, avg_pow):
        u_avg = 2*self.chain.l_phi(average_key).data
        u_avg = u_avg - avg_pow.data
        u_avg_ind = cp.asnumpy(self.xp.argsort(u_avg, axis=1))
        phi_ind = np.zeros(self.nb_class_test)
        for i in range(self.nb_class_test):
            if i == 0:
                phi_ind[i] = np.int(u_avg_ind[i, self.nb_class_train-1])
            else:
                k=self.nb_class_train-1
                while u_avg_ind[i,k] in phi_ind[:i]:
                    k = k-1
                phi_ind[i] = np.int(u_avg_ind[i,k])
        return phi_ind.tolist()
        
    def train(self, images, labels):
        """
        Train a minibatch of episodes
        """
        images = self.xp.stack(images)
        batchsize = images.shape[0]
        loss = 0

        key = self.encoder(images, batchsize, train=True)
        support_set = key[:self.nb_class_train*self.n_shot,:]
        query_set = key[self.nb_class_train*self.n_shot:,:]
        average_key = F.mean(F.reshape(support_set,[self.n_shot,self.nb_class_train,-1]),axis=0)
        
        batchsize_q = len(query_set.data)
        M = self.Projection_Space(average_key, batchsize_q, self.nb_class_train)

        r_t = F.reshape(F.batch_matmul(M,F.batch_matmul(M,query_set,transa=True)),(batchsize_q,-1))
        
        pow_t = self.compute_power(batchsize_q,query_set,M,self.nb_class_train)
        
        loss = self.compute_loss(labels[self.nb_class_train*self.n_shot:], r_t, pow_t, batchsize_q,self.nb_class_train)
                
        self.chain.zerograds()
        loss.backward()
        self.optimizer.update()
        
        return loss.data

        
    def evaluate(self, images, labels):
        """
        Evaluate accuracy score
        """
        nb_class = self.nb_class_test
            
        images = self.xp.stack(images)
        batchsize = images.shape[0]
        accs = []
        
        key= self.encoder(images,batchsize, train=False)
        support_set = key[:nb_class*self.n_shot,:]
        query_set = key[nb_class*self.n_shot:,:]
        average_key = F.mean(F.reshape(support_set,[self.n_shot,nb_class,-1]),axis=0)
        batchsize_q = len(query_set.data)
        pow_avg = self.compute_power_avg_phi(batchsize_q, nb_class, average_key, train=False)
        
        phi_ind = [np.int(ind) for ind in self.select_phi(average_key,pow_avg)]

        M = self.Projection_Space(average_key, batchsize_q,nb_class, train=False, phi_ind=phi_ind)
        r_t = F.reshape(F.batch_matmul(M,F.batch_matmul(M,query_set,transa=True)),(batchsize_q,-1))
        pow_t = self.compute_power(batchsize_q,query_set,M,nb_class, train=False, phi_ind=phi_ind)
        
        accs_tmp = self.compute_accuracy(labels[nb_class*self.n_shot:], r_t, pow_t, batchsize_q, nb_class, phi_ind=phi_ind)
        
        accs.append(accs_tmp)

                
        return accs
    

    
    def decay_learning_rate(self, decaying_parameter=0.5):
        self.optimizer.alpha=self.optimizer.alpha*decaying_parameter
        