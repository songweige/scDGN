import torch.nn as nn
from .function_util import ReverseLayerF
        
class ClassicNN(nn.Module):
    def __init__(self, d_dim=20499, dim1=1136, dim2=100, l_dim=21):
        super(ClassicNN, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.d_dim = d_dim
        self.l_dim = l_dim
        self.h1 = nn.Sequential(
           nn.Linear(self.d_dim, self.dim1),
           nn.Tanh(),
        )
        self.h2 = nn.Sequential(
           nn.Linear(self.dim1, self.dim2),
           nn.Tanh(),
        )
        self.o = nn.Sequential(
            nn.Linear(self.dim2, self.l_dim),
        )
        print(self)

    def forward(self, x, mode='train'):
        h1_output = self.h1(x)
        h2_output = self.h2(h1_output)
        class_output = self.o(h2_output)
        if mode ==  'train' or 'test':
            return class_output
        elif mode == 'eval':
            return h2_output

class scDGN(nn.Module):
    def __init__(self, d_dim=20499, dim1=1136, dim2=100, dim_label=75, dim_domain=64):
        super(scDGN, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.d_dim = d_dim
        self.dim_label = dim_label
        self.dim_domain = dim_domain
        self.feature_extractor = nn.Sequential(
           nn.Linear(self.d_dim, self.dim1),
           nn.Tanh(),
           nn.Linear(self.dim1, self.dim2),
           nn.Tanh(),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.dim2, self.dim_domain),         
            nn.Tanh(),
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(self.dim2, self.dim_label),
        )
        print(self)
    def forward(self, x1, x2=None, mode='train', alpha=1):
        feature = self.feature_extractor(x1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.label_classifier(feature)
        if mode == 'train':
            domain_output1 = self.domain_classifier(reverse_feature)
            feature2 = self.feature_extractor(x2)
            reverse_feature2 = ReverseLayerF.apply(feature2, alpha)
            domain_output2 = self.domain_classifier(reverse_feature2)
            return class_output, domain_output1, domain_output2
        elif mode == 'test':
            return class_output
        elif mode == 'eval':
            return feature