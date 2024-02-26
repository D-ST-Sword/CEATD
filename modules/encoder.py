import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.align import *

class fineTextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(fineTextEncoder, self).__init__()
        
        self.lstm = nn.LSTM(embedding_dim, output_dim//2, num_layers=3, bidirectional=True,dropout=0.85)
        
    def forward(self, text):  
     
        lstm_out, _ = self.lstm(text)  
  
        return lstm_out[:, -1, :]

class coarseTextEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(coarseTextEncoder, self).__init__()
        
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=4)
        
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, text):  
        
        transformer_out = self.transformer_encoder(text)  
        
        
        transformer_out = self.layer_norm(transformer_out)
        
        return  transformer_out[:, -1, :] 
    
class fineAudioEncoder(nn.Module):
    def __init__(self, output_dim):
        super(fineAudioEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

    def forward(self, audio):   
        audio = audio.squeeze(0)
        audio = F.relu(self.bn1(self.conv1(audio)))
        audio = self.dropout(audio)
        audio = F.relu(self.bn2(self.conv2(audio)))
        fine_grained = self.pool1(audio)
        return fine_grained
    
class coarseAudioEncoder(nn.Module):
    def __init__(self, output_dim):
        super(coarseAudioEncoder, self).__init__()
        self.biLSTM = nn.LSTM(74, output_dim , num_layers=5, bidirectional=True,dropout=0.85    )  
        

    def forward(self, audio):   
        audio = audio.squeeze(0)      
        lstm_out, _ = self.biLSTM(audio) 
        coarse_features = lstm_out[:, -1, :]  
        
        return  coarse_features 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

class fineVideoEncoder(nn.Module):
    def __init__(self, output_dim):
        super(fineVideoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.residual_block1 = ResidualBlock(32)
        self.residual_block2 = ResidualBlock(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, video):      
        out = F.relu(self.conv1(video))
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        fine_features = self.pool1(out)
        return fine_features
    
class coarseVideoEncoder(nn.Module):
    def __init__(self, output_dim):
        super(coarseVideoEncoder, self).__init__()
        
        self.biLSTM = nn.LSTM(35, output_dim, num_layers=3, bidirectional=True,dropout=0.85)  
         
    def forward(self, video):  
        
        video = video.squeeze()  
        
        coarse_features = self.biLSTM(video)[0][:, -1, :]
        
        return  coarse_features 
'''
class fineTextEncoder_contrast(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, common_dim):
        super(fineTextEncoder_contrast, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Add if needed
        self.lstm = nn.LSTM(embedding_dim, output_dim//2, num_layers=2, bidirectional=True)
        self.projection_head = projectionHead(output_dim, common_dim)

    def forward(self, text):  
        #embedded = self.embedding(text)  # Uncomment if using embeddings
        lstm_out, _ = self.lstm(text)  
        embedding = self.projection_head(lstm_out[:, -1, :]) 
        return embedding 
    
class coarseTextEncoder_contrast(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, common_dim):
        super(coarseTextEncoder_contrast, self).__init__()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.projection_head = projectionHead(output_dim, common_dim)  

    def forward(self, text):  
        transformer_out = self.transformer_encoder(text)  
        embedding = self.projection_head(transformer_out[:, -1, :])  
        return  embedding 
'''