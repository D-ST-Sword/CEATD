import torch
import torch.nn as nn
import torch.nn.functional as F


class transpose2D(nn.Module):
    '''
    transpose all embeddings to [batch_size, setting height, setting weight]
    '''
    def __init__(self,exshape):
        super(transpose2D, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.exshape = exshape
    def forward(self, x):
        
        dims = len(x.shape)
        x = x.unsqueeze(0)
        

        
        if dims == 4:
            x = x.squeeze(0)
        elif dims < 3:
            for _ in range(3 - dims):
                x = x.unsqueeze(-1)
        elif dims > 3:
            
            # if the input is over 3D, we need to flatten the input
            x = x.view(x.shape[0], -1)

        
        # upsample the input to the desired shape
        while x.size(2) < self.exshape[0] or x.size(3) < self.exshape[1]:
            x = self.upconv(x)

        # downsample the input to the desired shape
        if x.size(2) > self.exshape[0] or x.size(3) > self.exshape[1]:
            x = x[:, :, :self.exshape[0], :self.exshape[1]]
        x = x.squeeze(0) 
        return x
    

'''
use mlp layet to align all the embeddings to the same size

'''

class projectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(projectionHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
'''
SAGU moduel
'''
class SAGU(nn.Module):
    def __init__(self, embedding_dim, output_dim, top_k):
        super(SAGU, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.top_k = top_k

        self.weight_fc_fine = nn.Linear(embedding_dim[0], 128) 
        self.weight_fc_coarse = nn.Linear(embedding_dim[1], 128)

        self.fine_linear = nn.Linear(embedding_dim[2], 128)
        self.coarse_linear = nn.Linear(embedding_dim[3], 128)

        self.fine_resizer = transpose2D([128,128])
        self.coarse_resizer = transpose2D([128,128])

    def forward(self, fine_embedding, coarse_embedding):
        fine_embedding =self.fine_resizer(fine_embedding)
        coarse_embedding = self.coarse_resizer(coarse_embedding)
        
        fine_weights = torch.sigmoid(self.weight_fc_fine(fine_embedding)) 
        coarse_weights = torch.sigmoid(self.weight_fc_coarse(coarse_embedding)) 

        _, fine_topk_indices = torch.topk(fine_weights, self.top_k, dim=-1)
        _, coarse_topk_indices = torch.topk(coarse_weights, self.top_k, dim=-1) 

        fine_selected = torch.gather(fine_embedding, dim=-1, index=fine_topk_indices)
        coarse_selected = torch.gather(coarse_embedding, dim=-1, index=coarse_topk_indices)

        aligned_fine = F.pad(fine_selected, (0, self.output_dim - self.top_k))
        aligned_coarse = F.pad(coarse_selected, (0, self.output_dim - self.top_k))

        aligned_fine = self.fine_linear(aligned_fine)
        
        aligned_coarse = self.coarse_linear(aligned_coarse)
        
        return aligned_fine, aligned_coarse