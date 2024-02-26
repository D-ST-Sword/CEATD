import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sklearn.manifold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from ..CEATD.model.net import *
from ..CEATD.modules.encoder import *
import time
from torch.utils.data import random_split
from torchsummary import summary
from sklearn.metrics import f1_score
import torch.nn.functional as F  
from ..CEATD.train.train import train_model, contrast_train

dataPath = '' # path to the data

loader= torch.load(dataPath)

dataset_size = len(loader)
train_size = int(0.8 * dataset_size)  # traning set takes 80%
test_size = dataset_size - train_size  # test set takes 20%
# set random seed
torch.manual_seed(1)

train_dataset, test_dataset = random_split(loader, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)
'''
# Hyperparameters for demo
embedding_dim = 300  # Example word embedding dimension
output_dim =  128    # Output dimension of encoder and  fused representations
sentiment_classes = 7  # Assuming 7 output classes
k = 10                # Parameter for SAGU
input_dim = 300       # Input dimension for the text encoder

# Create model instance
basemodel = baseMultiModel(fineTextEncoder(input_dim, embedding_dim=embedding_dim, output_dim=output_dim),
                        fineAudioEncoder(output_dim=output_dim),
                        fineVideoEncoder(output_dim=output_dim),
                        coarseTextEncoder(input_dim, embedding_dim=embedding_dim, output_dim=output_dim),
                        coarseAudioEncoder(output_dim=output_dim),
                        coarseVideoEncoder(output_dim=output_dim),
                        output_dim=sentiment_classes,
                        k=k)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = basemodel.to(device)
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
# Train the basemodel
train_model(basemodel, train_loader, test_loader, device, Loss, optimizer, epochs)


#use contrast model


contrastmodel = contrastMultiModel(fineTextEncoder(input_dim, embedding_dim=embedding_dim, output_dim=output_dim),
                        fineAudioEncoder(output_dim=output_dim),
                        fineVideoEncoder(output_dim=output_dim),
                        coarseTextEncoder(input_dim, embedding_dim=embedding_dim, output_dim=output_dim),
                        coarseAudioEncoder(output_dim=output_dim),
                        coarseVideoEncoder(output_dim=output_dim),
                        output_dim=sentiment_classes,
                        k=k)

# Train the contrastmodel
contrast_train(contrastmodel, train_loader, optimizer, Loss)
'''