import os
import numpy as np
from tqdm import tqdm, trange

class dataset(object):
    def __init__(self, data_path, batch_size, label_size=46, dataset_name=None, validation=True):
        super(dataset, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.label_size = label_size
        self.dataset_name = dataset_name
        self.validation = validation
        self.init_dataset()
    def init_dataset(self):
        print('loading dataset: %s'%self.dataset_name)
        if self.dataset_name == 'scquery':
            self.train_set = np.load(os.path.join(self.data_path, 'train_scquery.npz'))
            self.valid_set = np.load(os.path.join(self.data_path, 'valid_scquery.npz'))
            self.test_set = np.load(os.path.join(self.data_path, 'test_scquery.npz'))
        elif self.dataset_name.startswith('pancreas'):
            data_file = 'pancreas%s.npz'%self.dataset_name[-1]
            all_set = np.load(os.path.join(self.data_path, data_file))
            self.train_set = {'features':all_set['features'][:-638], 'labels': all_set['labels'][:-638], 'accessions':all_set['accessions'][:-638]}
            self.test_set = {'features':all_set['features'][-638:], 'labels': all_set['labels'][-638:], 'accessions':all_set['accessions'][-638:]}
        elif self.dataset_name == 'pbmc':
            all_set = np.load(os.path.join(self.data_path, 'pbmc.npz'))
            self.train_set = {'features':all_set['features'][2992:], 'labels': all_set['labels'][2992:], 'accessions':all_set['accessions'][2992:]}
            self.test_set = {'features':all_set['features'][:2992], 'labels': all_set['labels'][:2992], 'accessions':all_set['accessions'][:2992]}
        else:
            print('Wrong name, cannot find the dataset.')  
            
        self._test_X = self.test_set['features']
        self._test_y = self.test_set['labels']
        self.accessions_set =list(set(self.train_set['accessions']))
        if 'valid_set' in vars(self).keys():
            self._train_X = self.train_set['features']
            self._train_y = self.train_set['labels']
            self._valid_X = self.valid_set['features']
            self._valid_y = self.valid_set['labels']
            self._train_acc = np.array([self.accessions_set.index(item) for item in list(self.train_set['accessions'])])
            self.n_valid = self.valid_set['features'].shape[0]
            self.num_valid_batches = len(self._valid_X)//self.batch_size
        elif self.validation: # randomly extract 20% of training set as validation set
            self.n_sample = int(self.train_set['features'].shape[0])
            self.n_valid = int(0.2*self.n_sample )
            self.perm = np.random.permutation(self.n_sample)
            self._all_X = self.train_set['features']
            self._all_y = self.train_set['labels']
            self._all_acc = np.array([self.accessions_set.index(item) for item in list(self.train_set['accessions'])])
            self._train_X = self._all_X[self.perm[:-self.n_valid]]
            self._valid_X = self._all_X[self.perm[-self.n_valid:]]
            self._train_acc = self._all_acc[self.perm[:-self.n_valid]]
            self._valid_acc = self._all_acc[self.perm[-self.n_valid:]]
            self._train_y = self._all_y[self.perm[:-self.n_valid]]
            self._valid_y = self._all_y[self.perm[-self.n_valid:]]
            self.num_valid_batches = len(self._valid_X)//self.batch_size
        else:
            self._train_X = self.train_set['features']
            self._train_y = self.train_set['labels']
            self._train_acc = np.array([self.accessions_set.index(item) for item in list(self.train_set['accessions'])])

        self.num_train_batches = len(self._train_X)//self.batch_size
        self.num_test_batches = len(self._test_X)//self.batch_size
        self.labels_set = np.unique(self._train_y)
        self.label_to_indices = {label: np.where(self._train_y == label)[0]
                                    for label in self.labels_set}
        
    def train_data(self, loss=0.0):
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_X)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_y)
        return tqdm(self.next_train_batch(),
                    desc='Train loss: {:.4f}'.format(loss),
                    total=self.num_train_batches, mininterval=1.0, leave=False)

    def next_train_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._train_X)
        while start < N:
            if self.validation:
                valid_start = np.random.randint(0, self.n_valid-self.batch_size)
                valid_end = self.batch_size + valid_start
            if end > N:
                if self.validation:
                    yield self._train_X[start:], self._train_y[start:], \
                        self._valid_X[valid_start:valid_end], self._valid_y[valid_start:valid_end]
                else:
                    yield self._train_X[start:], self._train_y[start:], None, None
            else:
                if self.validation:
                    yield self._train_X[start:end], self._train_y[start:end], \
                        self._valid_X[valid_start:valid_end], self._valid_y[valid_start:valid_end]
                else:
                    yield self._train_X[start:end], self._train_y[start:end], None, None
            start = end
            end += self.batch_size

    def test_data(self):
        return tqdm(self.next_test_batch(), desc='Test Iterations: ',
                    total=self.num_test_batches, leave=False )

    def next_test_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._test_X)
        while start < N:
            if end > N:
                yield self._test_X[start:], self._test_y[start:]
            else:                
                yield self._test_X[start:end], self._test_y[start:end]
            start = end
            end += self.batch_size

    def valid_data(self):
        return tqdm(self.next_valid_batch(), desc='Valid Iterations: ',
                    total=self.num_valid_batches, leave=False )

    def next_valid_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._valid_X)
        while start < N:
            if end > N:
                yield self._valid_X[start:end], self._valid_y[start:end]
            else:                
                yield self._valid_X[start:end], self._valid_y[start:end]
            start = end
            end += self.batch_size        

    def compute_MI(self, x, y):
        x = self._train_y
        y = self._train_acc
        sum_mi = 0.0
        x_value_list = np.unique(x)
        y_value_list = np.unique(y)
        Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
        Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
        for i in range(len(x_value_list)):
            if Px[i] ==0.:
                continue
            sy = y[x == x_value_list[i]]
            if len(sy)== 0:
                continue
            pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
            t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
            sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
        return sum_mi, scipy.stats.entropy(Px, base=2), scipy.stats.entropy(Py, base=2)

class dataset_DG(dataset):
    def __init__(self, data_path, batch_size, label_size, dataset_name=None, validation=True):
        super(dataset_DG, self).__init__(data_path, batch_size, label_size, dataset_name, validation)
        self.prepare_siamese()
        
    def prepare_siamese(self):
        raise NotImplementedError
    
    def train_data(self, loss=0.0):
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_X)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_X2)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_y)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_z)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_acc)
        np.random.set_state(rng_state)
        np.random.shuffle(self._use_domain)
        return tqdm(self.next_train_batch(),
                    desc='Train loss: {:.4f}'.format(loss),
                    total=self.num_train_batches, mininterval=1.0, leave=False)
    
    def next_train_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._train_X)
        while start < N:
            if self.validation:
                valid_start = np.random.randint(0, self.n_valid-self.batch_size)
                valid_end = self.batch_size + valid_start
            if end < N:
                if self.validation:
                    yield self._train_X[start:end], self._train_X2[start:end], self._train_y[start:end],\
                        self._train_z[start:end], self._use_domain[start:end], \
                        self._valid_X[valid_start:valid_end], self._valid_y[valid_start:valid_end]
                else:
                    yield self._train_X[start:end], self._train_X2[start:end], self._train_y[start:end],\
                        self._train_z[start:end], self._use_domain[start:end], None, None
            else:
                if self.validation:
                    yield self._train_X[start:], self._train_X2[start:], self._train_y[start:],\
                        self._train_z[start:], self._use_domain[start:], \
                        self._valid_X[valid_start:valid_end], self._valid_y[valid_start:valid_end]
                else:
                    yield self._train_X[start:], self._train_X2[start:], self._train_y[start:],\
                        self._train_z[start:], self._use_domain[start:], None, None
            start = end
            end += self.batch_size
            
class dataset_ADG(dataset_DG):
    """docstring for dataset, need indicate data_path and batch_size"""
    def __init__(self, data_path, batch_size, label_size, dataset_name=None, validation=True):
        super(dataset_ADG, self).__init__(data_path, batch_size, label_size, dataset_name, validation)
    
    def prepare_siamese(self):
        self._train_X2 = np.zeros(self._train_X.shape)
        self._train_z = np.zeros(self._train_acc.shape)
        self._use_domain = np.ones(self._train_acc.shape)
        for index in range(self._train_X.shape[0]):
            label = self._train_y[index]
            accession = self._train_acc[index]
            acc_positive_ids = np.where(self._train_acc==accession)[0]
            acc_negative_ids = np.where(self._train_acc!=accession)[0]
            if np.random.random()>0.5 or len(acc_negative_ids)<=1:
                siamese_index = np.random.choice(acc_positive_ids)
                self._train_X2[index] = self._train_X[siamese_index]
                self._train_z[index] = 1
            else:
                siamese_index = np.random.choice(acc_negative_ids)
                self._train_X2[index] = self._train_X[siamese_index]
                self._train_z[index] = 0
            

class dataset_CADG(dataset_DG):
    def __init__(self, data_path, batch_size, label_size, dataset_name=None, validation=True):
        super(dataset_CADG, self).__init__(data_path, batch_size, label_size, dataset_name, validation)
    
    def prepare_siamese(self):
        self._train_X2 = np.zeros(self._train_X.shape)
        self._train_z = np.zeros(self._train_acc.shape, dtype=int)
        self._use_domain = np.ones(self._train_acc.shape) # not using domain info if the label and domain both occur once
        for index in range(self._train_X.shape[0]):
            label = self._train_y[index]
            accession = self._train_acc[index]
            acc_positive_ids = np.where(self._train_acc==accession)[0]
            acc_negative_ids = np.where(self._train_acc!=accession)[0]
            positive_ids = np.setdiff1d(acc_positive_ids, self.label_to_indices[label])
            negative_ids = np.intersect1d(acc_negative_ids, self.label_to_indices[label])
            if np.random.random()>0.5 and len(positive_ids) >= 1:
                siamese_index = np.random.choice(positive_ids)
                self._train_X2[index] = self._train_X[siamese_index]
                self._train_z[index] = 1
            elif len(negative_ids) >= 1:
                siamese_index = np.random.choice(negative_ids)
                self._train_X2[index] = self._train_X[siamese_index]
                self._train_z[index] = 0
            else:
                self._use_domain[index] = 0
                
    def get_domain_info(self):
        self.domain2label = {i:[] for i in range(len(set(self._train_acc)))}
        self.label2domain = {i:[] for i in range(self.label_size)}        
        for i in range(self._train_acc.shape[0]):
            if self._train_y[i] not in self.domain2label[self._train_acc[i]]:
                self.domain2label[self._train_acc[i]].append(self._train_y[i])
        for i in range(self._train_acc.shape[0]):
            if self._train_acc[i] not in self.label2domain[self._train_y[i]]:
                self.label2domain[self._train_y[i]].append(self._train_acc[i])
        return self.domain2label, self.label2domain

    def plot_bipartitle(self):
        n_domain = len(self.domain2label)
        n_label = len(self.label2domain)
        for i in range(n_domain):
            y_domain = i/n_domain
            for label in self.domain2label[i]:
                y_label = label/n_label
                plt.plot([0, 1], [y_domain, y_label], color='k', linewidth=1, marker='.', markerfacecolor='red', markersize=8) 

        plt.xticks([0, 1], ['domain', 'label'])
        plt.yticks([])