import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn import *
import scipy
import time
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping, Callback

class FeatureNN(torch.nn.Module):
    def __init__(self, n_layers: tuple = (int), layer_activation = torch.nn.ReLU(), p_dropout = 0.5):
        super().__init__()
        
        self.n_layers = n_layers
        self.layer_activation = layer_activation
        self.p_dropout = p_dropout
        self.layer_dropout = torch.nn.Dropout(p = p_dropout)
        self.layers=[]
        
        prev = 1
        
        for n_layer in n_layers:
            self.layers.append(torch.nn.Linear(prev,n_layer))
            self.layers.append(self.layer_dropout)
            self.layers.append(self.layer_activation)
            prev = n_layer
        # last layer
        self.layers.append(torch.nn.Linear(prev,1))

        # convert to modulelist
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.unsqueeze(1)
    
class NAM(torch.nn.Module):
    def __init__(self, input_size, n_layers: tuple = (int), layer_activation = torch.nn.ReLU(), p_dropout = 0.5):
        super().__init__()
        self.input_size = input_size
        
        self.FeatureNNs = torch.nn.ModuleList([
            FeatureNN(n_layers = n_layers, layer_activation = layer_activation, p_dropout = p_dropout)
            for i in range(self.input_size)
        ])
        
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        # individual contributions
        contrib = torch.cat(self.featurewise(x), dim=-1)
        # sum + bias, need 
        x = contrib.sum(axis=-1) + self.bias
        x = self.sigmoid(x)
        x = torch.cat([1-x,x],axis=1) # [:,1] = real pred, did this to satisfy predict_proba which requires two columns even for binary classification
        return x
        #return {'y_hat':x,'contrib':contrib}
        
    def contrib(self,x):
        return torch.cat(self.featurewise(x), dim=-1)
    
    # stolen from https://github.com/kherud/neural-additive-models-pt/blob/master/nam/model.py
    def featurewise(self, x):
        # different featureNN for each feature
        return [self.FeatureNNs[i](x[:, i]) for i in range(self.input_size)]

# [:,1] = real pred, did this to satisfy predict_proba which requires two columns even for binary classification
# skorch expects a loss function that needs to be initialized, thus the double wrapper
def customloss(lossfunc):
    def outerdummyfunc():
        def innerdummyfunc(y_hat,y):
            return lossfunc(y_hat[:,[1]],y)
        return innerdummyfunc
    return outerdummyfunc
# callback for plotting training regimen after finishing
class TrainPlot(Callback):
    def __init__(self, metrictuples):
        self.metrictuples = metrictuples # tuple of tuples, grouped by subplot

    def initialize(self):
        #If you have attributes that should be reset when the model is re-initialized, those attributes should be set in this method.
        return

    def on_train_end(self, net, **kwargs):
        #display(kwargs) # what's in here? {'x':data, 'y':data}
        
        nrows = len(self.metrictuples) + 1
        fig,axs = plt.subplots(figsize=(24, 2 * nrows), nrows = nrows, sharex=True)
        history = pd.DataFrame(net.history)
        
        for axi,metrictuple in enumerate(self.metrictuples):
            for metric in metrictuple:
                
                color = 'black'
                if 'valid' in metric.lower():
                    color='red'
                
                axs[axi].plot(history['epoch'],history[metric],label=metric, color=color)
                
                if 'loss' in metric.lower():
                    axs[axi].axvline(history['epoch'].iloc[history[metric].idxmin()],linestyle=':', color=color)
                elif metric=='dur':
                    continue
                else:
                    axs[axi].axvline(history['epoch'].iloc[history[metric].idxmax()],linestyle=':', color=color)
            axs[axi].legend(bbox_to_anchor=(1.001,0.5), loc='center left')
            axs[axi].grid(axis='y')
        
        # improvement plot
        axi+=1
        cummin = history['valid_loss'].cummin()
        axs[axi].plot(history['epoch'],cummin, color='red', label='valid_loss')
        for shiftpoint in history.loc[cummin.loc[cummin!=cummin.shift()].index,'epoch'].values:
            axs[axi].axvline(shiftpoint,color='red',linestyle=':')
        axs[axi].legend(bbox_to_anchor=(1.001,0.5), loc='center left')
        axs[axi].grid(axis='y')
        
        axs[axi].set_xticks(np.arange(1,history['epoch'].values[-1]+1,1))
        axs[axi].set_xlabel('Epochs')
        plt.subplots_adjust(hspace=0.1)
        plt.show()
class NAMClassifier(skorch.NeuralNetClassifier):
    def __init__(self, input_size, n_layers = (16,16,16), layer_activation = torch.nn.ReLU, p_dropout = 0.5, max_epochs = 1000, batch_size=2**10, cv = 5, optimizer = torch.optim.Adam, lr = 0.003, es=True):
        
        # generate callbacks
        callbacks = [
            skorch.callbacks.EpochScoring(name='train_AUROC',scoring='roc_auc',           lower_is_better=False, on_train=True),
            skorch.callbacks.EpochScoring(name='train_AUPRC',scoring='average_precision', lower_is_better=False, on_train=True),
            skorch.callbacks.EpochScoring(name='valid_AUROC',scoring='roc_auc',           lower_is_better=False, on_train=False),
            skorch.callbacks.EpochScoring(name='valid_AUPRC',scoring='average_precision', lower_is_better=False, on_train=False),
            TrainPlot(metrictuples=(
                ('train_loss','valid_loss'),
                ('train_AUROC','valid_AUROC'),
                ('train_AUPRC','valid_AUPRC'),
                ('dur',),
            )),
        ]
        if es:
            callbacks.append(skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=20, threshold=1e-4, threshold_mode='rel', lower_is_better=True)) #https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.EarlyStopping
        
        # call init
        super().__init__(
            # neural network 
            module = NAM,
            module__input_size = input_size,
            module__n_layers = n_layers,
            module__layer_activation = layer_activation(),
            module__p_dropout = p_dropout,

            # training
            max_epochs = max_epochs,
            batch_size = batch_size,
            train_split = skorch.dataset.ValidSplit(cv=cv, stratified=True), # similar to sklearn cv, but only splits once: https://skorch.readthedocs.io/en/stable/user/dataset.html?highlight=validsplit#validsplit

            # optimizer
            optimizer = optimizer,
            optimizer__lr=lr,

            # criterion
            criterion = customloss(torch.nn.BCELoss()), #NLLLoss

            # callback
            callbacks = callbacks,

            # misc
            warm_start = False,
            #device='cuda',  # uncomment this to train with CUDA
        )
    def transform(self, x):
        sklearn.utils.validation.check_is_fitted(self)
        ret = self.module_.contrib(torch.tensor(x)).detach().squeeze(1).numpy()
        return ret