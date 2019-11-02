import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .model_util import ClassicNN, scDGN
from .loss_util import ContrastiveLoss

class BaseTrainer(object):
    def __init__(self, d_dim, dim1, dim2, dim_label, num_epochs, batch_size, use_gpu, validation=True):
        super(BaseTrainer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim_label = dim_label
        self.d_dim = d_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.validation = validation
        
    def initialize(self):
        for p in self.D.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform(p)
            else:
                nn.init.normal(p, 0, 1)
    def test(self):
        self.D.eval()
        counts = 0
        sum_acc = 0.0
        for x, y in self.dataset.test_data():
            counts += len(y)
            if self.use_gpu:
                X = Variable(torch.cuda.FloatTensor(x))
                Y = Variable(torch.cuda.LongTensor(y))
            else:
                X = Variable(torch.FloatTensor(x))
                Y = Variable(torch.LongTensor(y))
            outputs = self.D(X, X)
            f_X = outputs[0]
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('Test Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc
      
    def valid(self):
        self.D.eval()
        counts = 0
        sum_acc = 0.0
        for x, y in self.dataset.valid_data():
            counts += len(y)
            if self.use_gpu:
                X = Variable(torch.cuda.FloatTensor(x))
                Y = Variable(torch.cuda.LongTensor(y))
            else:
                X = Variable(torch.FloatTensor(x))
                Y = Variable(torch.LongTensor(y))
            outputs = self.D(X, X)
            f_X = outputs[0]
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
            del X,Y,f_X
        valid_acc = sum_acc/counts
        print('Valid Accuracy: {0:0.3f}'.format(valid_acc))
        return valid_acc

class ClassicTrainer(BaseTrainer):
    def __init__(self, d_dim, dim1, dim2, dim_label, num_epochs, batch_size, use_gpu, validation=True):
        super(ClassicTrainer, self).__init__(d_dim, dim1, dim2, dim_label, num_epochs, batch_size, use_gpu, validation)
        if self.use_gpu:
            self.D = ClassicNN(self.d_dim, self.dim1, self.dim2, self.dim_label).cuda()
            self.L = nn.CrossEntropyLoss().cuda()
        else:
            self.D = ClassicNN(self.d_dim, self.dim1, self.dim2, self.dim_label)
            self.L = nn.CrossEntropyLoss()
            
        self.optimizer = optim.SGD([{'params':self.D.parameters()}], lr=1e-3, momentum=0.9, weight_decay=1e-6, nesterov=True)

    def train(self, f):
        self.D.train()
        self.initialize()
        loss_val = float('inf')
        self.train_loss = []
        self.valid_accs = []
        best_validate_acc = 0.0
        for j in range(self.num_epochs):
            begin = time.time()
            counts = 0
            sum_acc = 0.0
            valid_sum_acc = 0.0
            train_epoch_loss = []
            valid_epoch_loss = []
            train_data = self.dataset.train_data(float(loss_val))
            for x, y, self.x_valid, y_valid in train_data:
                # forward calculation and back propagation
                counts += len(y)
                if self.use_gpu:
                    X = Variable(torch.cuda.FloatTensor(x))
                    Y = Variable(torch.cuda.LongTensor(y))
                else:
                    X = Variable(torch.FloatTensor(x))
                    Y = Variable(torch.LongTensor(y))
                self.optimizer.zero_grad()
                f_X = self.D(X)
                loss = self.L(f_X, Y)
                loss_val = loss.data.cpu().numpy()
                sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
                train_data.set_description('Train loss: {0:.4f}'.format(float(loss_val)))
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss_val)
                del X,Y,f_X,loss
                # calculate validation loss
                if self.validation:
                    if self.use_gpu:
                        X = Variable(torch.cuda.FloatTensor(self.x_valid))
                        Y = Variable(torch.cuda.LongTensor(y_valid))
                    else:
                        X = Variable(torch.FloatTensor(self.x_valid))
                        Y = Variable(torch.LongTensor(y_valid))
                    outputs = self.D(X)
                    f_X = outputs[0]
                    loss = self.L(f_X, Y)
                    loss_val = loss.data.cpu().numpy()
                    valid_sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
                    valid_epoch_loss.append(loss_val)
                    del X,Y,f_X

            self.train_acc = sum_acc/counts
            self.train_loss.append(np.sum(train_epoch_loss)/(counts/self.batch_size))
            if self.validation:
                valid_acc = valid_sum_acc/counts
                self.valid_accs.append(valid_acc)
                f.write("Epoch %d, time = %ds, train accuracy = %.4f, loss = %.4f, validation accuracy = %.4f\n" % (
                        j, time.time() - begin, self.train_acc, self.train_loss[-1], valid_acc))
                if self.valid_accs[-1] > best_validate_acc:
                    best_validate_acc = self.valid_accs[-1]
                    score = self.test()
                    f.write("Best Validated Model Prediction Accuracy = %.4f\n" % (score))
            else:
                f.write("Epoch %d, time = %ds, train accuracy = %.4f, loss = %.4f, test accuracy = %.4f\n" % (
                        j, time.time() - begin, self.train_acc, self.train_loss[-1], self.test()))
            if j%10==9:
                tqdm.write('After epoch {0} Train Accuracy: {1:0.3f} '.format(j+1, self.train_acc))
        if self.validation:
            f.write("Best Validated Model Prediction Accuracy = %.4f\n" % (score))
        f.write('After Training, Test Accuracy: {:0.3f}\n'.format(self.test()))


class ADGTrainer(BaseTrainer):
    def __init__(self, d_dim, margin, lamb, dim1, dim2, dim_label, dim_domain, num_epochs, batch_size, use_gpu=True, validation=True):
        #Setup network
        super(ADGTrainer, self).__init__(d_dim, dim1, dim2, dim_label, num_epochs, batch_size, use_gpu, validation)
        self.lamb = lamb
        self.dim_domain = dim_domain
        self.D = DANN_Siamese(self.d_dim, self.dim1, self.dim2, self.dim_label, self.dim_domain).cuda()
        self.L_L = nn.CrossEntropyLoss().cuda()
        self.L_D = ContrastiveLoss(margin=margin).cuda()
        self.optimizer = optim.SGD([{'params':self.D.parameters()}], lr=1e-3, momentum=0.9, weight_decay=1e-6, nesterov=True)
    def train(self, f):
        self.D.train()
        self.initialize()
        loss_val = float('inf')
        self.train_loss = []
        self.adv_loss = []
        self.valid_accs = []
        best_validate_acc = 0.0
        for j in range(self.num_epochs):
            begin = time.time()
            counts = 0
            sum_acc = 0.0
            valid_sum_acc = 0.0
            train_epoch_loss = []
            adv_epoch_loss = []
            valid_epoch_loss = []
            train_data = self.dataset.train_data(float(loss_val))
            for x1, x2, y, z, u, x_valid1, y_valid in train_data:
                #forward calculation and back propagation
                counts += len(y)
                X1 = Variable(torch.cuda.FloatTensor(x1))
                X2 = Variable(torch.cuda.FloatTensor(x2))
                Y = Variable(torch.cuda.LongTensor(y))
                Z = Variable(torch.cuda.LongTensor(z))
                U = Variable(torch.cuda.FloatTensor(u))
                self.optimizer.zero_grad()
                # label_output, domain_output = self.D(X1, X2)
                label_output, domain_output1, domain_output2 = self.D(X1, X2)
                label_loss = self.L_L(label_output, Y)
                domain_loss = self.L_D(domain_output1, domain_output2, Z, U)
                self.loss = label_loss + self.lamb*domain_loss
                label_loss_val = label_loss.data.cpu().numpy()
                domain_loss_val = domain_loss.data.cpu().numpy()
                sum_acc += (label_output.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
                train_data.set_description('Train label loss: {0:.4f}'.format(float(label_loss_val))+', domain loss: {0:.4f}'.format(float(domain_loss_val)))
                self.loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(label_loss_val)
                adv_epoch_loss.append(domain_loss_val)
                del X1, X2, Y, Z, U, label_output, domain_output1, domain_output2, label_loss
                #calculate validation loss
                if self.validation:
                    X_v = Variable(torch.cuda.FloatTensor(x_valid1))
                    Y_v = Variable(torch.cuda.LongTensor(y_valid))
                    label_output, _, _ = self.D(X_v, X_v)
                    loss = self.L_L(label_output, Y_v)
                    loss_val = loss.data.cpu().numpy()
                    valid_epoch_loss.append(loss_val)
                    valid_sum_acc += (label_output.max(dim=1)[1] == Y_v).float().sum().data.cpu().numpy()
                    del X_v,Y_v,label_output
            self.train_acc = sum_acc/counts
            self.train_loss.append(np.sum(train_epoch_loss)/(counts/self.batch_size))
            self.adv_loss.append(np.sum(adv_epoch_loss)/(counts/self.batch_size))
            if self.validation:
                valid_acc = valid_sum_acc/counts
                self.valid_accs.append(valid_acc)
                f.write("Epoch %d, time = %ds, train accuracy = %.4f, train loss = %.4f, adv loss = %.4f, validation accuracy = %.4f\n" % (
                        j, time.time() - begin, self.train_acc, self.train_loss[-1], self.adv_loss[-1], valid_acc))
                if self.valid_accs[-1] > best_validate_acc:
                    best_validate_acc = self.valid_accs[-1]
                    score = self.test()
                    f.write("Best Validated Model Prediction Accuracy = %.4f\n" % (score))
            else:
                f.write("Epoch %d, time = %ds, train accuracy = %.4f, train loss = %.4f, adv loss = %.4f, test accuracy = %.4f\n" % (
                        j, time.time() - begin, self.train_acc, self.train_loss[-1], self.adv_loss[-1], self.test()))
            if j%10==9:
                tqdm.write('After epoch {0} Train Accuracy: {1:0.3f}'.format(j+1, self.train_acc))
        if self.validation:
            f.write("Best Validated Model Prediction Accuracy = %.4f\n" % (score))
        f.write('After Training, Test Accuracy: {:0.3f}\n'.format(self.test()))

