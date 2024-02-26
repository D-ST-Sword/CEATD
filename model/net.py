from CEATD.modules.align import *
from CEATD.modules.shifting import *
from CEATD.modules.encoder import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class baseMultiModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(baseMultiModel, self).__init__()
        self.text_fine_encoder = fineTextEncoder(input_dim, embedding_dim, output_dim)
        self.audio_fine_encoder = fineAudioEncoder(output_dim)
        self.video_fine_encoder = fineVideoEncoder(output_dim)

        self.text_coarse_encoder = coarseTextEncoder(input_dim, embedding_dim, output_dim)
        self.audio_coarse_encoder = coarseAudioEncoder(output_dim)
        self.video_coarse_encoder = coarseVideoEncoder(output_dim)

        self.shifting = ATD_shifting()
        # SAGU or projection head
        # self.projectionHead = projectionHead(output_dim, output_dim)
        self.text_SAGU = SAGU(output_dim, output_dim, 8)
        self.audio_SAGU = SAGU(output_dim, output_dim, 8)
        self.video_SAGU = SAGU(output_dim, output_dim, 8)

        self.classifier = nn.Sequential(nn.Linear(770, 4096), 
                                        nn.GELU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 7)
                            )
    def forward(self, text, audio, video):
        fine_text = self.text_fine_encoder(text)
        fine_audio = self.audio_fine_encoder(audio)
        fine_video = self.video_fine_encoder(video)
        
        coarse_text = self.text_coarse_encoder(text)
        coarse_audio = self.audio_coarse_encoder(audio)
        coarse_video = self.video_coarse_encoder(video)
        

        text_refined, text_recoarse = self.text_SAGU(fine_text, coarse_text)
        
        audio_refined, audio_recoarse = self.audio_SAGU(fine_audio, coarse_audio)
        
        video_refined, video_recoarse = self.video_SAGU(fine_video, coarse_video)
        
        features_fusion = self.shifting(text_refined, audio_refined, video_refined, text_recoarse, audio_recoarse, video_recoarse)
        
        output = self.classifier(features_fusion)
        
        return output
    

class contrastMultiModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, common_dim):
        super(contrastMultiModel, self).__init__()
        self.text_fine_encoder = fineTextEncoder(input_dim, embedding_dim, output_dim)
        self.audio_fine_encoder = fineAudioEncoder(output_dim)
        self.video_fine_encoder = fineVideoEncoder(output_dim)

        self.text_coarse_encoder = coarseTextEncoder(input_dim, embedding_dim, output_dim)
        self.audio_coarse_encoder = coarseAudioEncoder(output_dim)
        self.video_coarse_encoder = coarseVideoEncoder(output_dim)

        self.shifting = ATD_shifting()
        # SAGU and projection head
        self.text_SAGU = SAGU(output_dim, output_dim, 8)
        self.audio_SAGU = SAGU(output_dim, output_dim, 8)
        self.video_SAGU = SAGU(output_dim, output_dim, 8)


        self.text_fine_projectionHead = projectionHead(output_dim, common_dim)
        self.text_coarse_projectionHead = projectionHead(output_dim, common_dim)
        self.audio_fine_projectionHead = projectionHead(output_dim, common_dim)
        self.audio_coarse_projectionHead = projectionHead(output_dim, common_dim)
        self.video_fine_projectionHead = projectionHead(output_dim, common_dim)
        self.video_coarse_projectionHead = projectionHead(output_dim, common_dim)


        # Contrastive Learning
        self.contrast_loss = nn.TripletMarginLoss()

        self.memory_bank = torch.randn(input_dim, requires_grad=False)

        self.classifier = nn.Sequential(nn.Linear(770, 4096), 
                                        nn.GELU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 7)
                            )
        
    def forward(self, text, audio, video):
        # Feature Extraction
        fine_text = self.text_fine_encoder(text)
        fine_audio = self.audio_fine_encoder(audio)
        fine_video = self.video_fine_encoder(video)
        
        coarse_text = self.text_coarse_encoder(text)
        coarse_audio = self.audio_coarse_encoder(audio)
        coarse_video = self.video_coarse_encoder(video) 

        #projection head
        fine_text = self.text_fine_projectionHead(fine_text)
        fine_audio = self.audio_fine_projectionHead(fine_audio)
        fine_video = self.video_fine_projectionHead(fine_video)
        
        coarse_text = self.text_coarse_projectionHead(coarse_text)
        coarse_audio = self.audio_coarse_projectionHead(coarse_audio)
        coarse_video = self.video_coarse_projectionHead(coarse_video)


        # Contrastive Learning (within-modality)
        loss_contrast = torch.tensor(0.0).cuda() 
        total_loss = torch.tensor(0.0).cuda() 
        for i in range(text.size(0)):
            modality = np.random.choice(['text', 'audio', 'video'])
            if modality == 'text':
                pos_idx = i  # Positive pair indices are the same
                neg_idx =  self.get_negative_example(i, 'text') 
                anchor = fine_text[i]
                positive = coarse_text[i]
                negative = fine_text[neg_idx] 
            loss_contrast += self.contrast_loss(anchor, positive, negative)
        for i in range(audio.size(0)):
            modality = np.random.choice(['text', 'audio', 'video'])
            if modality == 'audio':
                pos_idx = i
                neg_idx =  self.get_negative_example(i, 'audio')
                anchor = fine_audio[i]
                positive = coarse_audio[i]
                negative = fine_audio[neg_idx]
            loss_contrast += self.contrast_loss(anchor, positive, negative)
        for i in range(video.size(0)):
            modality = np.random.choice(['text', 'audio', 'video'])
            if modality == 'video':
                pos_idx = i
                neg_idx =  self.get_negative_example(i, 'video')
                anchor = fine_video[i]
                positive = coarse_video[i]
                negative = fine_video[neg_idx]

            loss_contrast += self.contrast_loss(anchor, positive, negative)
        total_loss += loss_contrast 

        text_refined, text_recoarse = self.text_SAGU(fine_text, coarse_text)
        
        audio_refined, audio_recoarse = self.audio_SAGU(fine_audio, coarse_audio)
        
        video_refined, video_recoarse = self.video_SAGU(fine_video, coarse_video)
        
        features_fusion = self.shifting(text_refined, audio_refined, video_refined, text_recoarse, audio_recoarse, video_recoarse)
        
        output = self.classifier(features_fusion)
        return output ,total_loss  

    def get_negative_example(self, idx, data,modality='text'):
        with torch.no_grad():
            fine_embed = self.text_fine_encoder(data[idx])  
            _, indices = self.memory_bank.topk(k=5, dim=0, largest=False)  # Find hard negatives (closest in embedding space)
            self.memory_bank[idx] = fine_embed  # Update the memory bank

        neg_idx = indices[np.random.randint(5)]  # Randomly select a hard negative 
        return neg_idx.item()  # Assuming neg_idx is a tensor
