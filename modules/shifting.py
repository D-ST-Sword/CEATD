import torch
import torch.nn as nn
import torch.nn.functional as F
from model.CEATD.modules.align import transpose2D

class ATD_shifting(nn.Module):
    def __init__(self):
        super(ATD_shifting, self).__init__()


        self.Conv2d_11 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,stride=1),
                                      nn.ReLU(),

                                      )
        self.Conv2d_12 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,stride=1),        

                                        nn.ReLU(),
                                            
                                            )
        self.Conv2d_13 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,stride=1,padding=0),
                                        nn.ReLU(),
                                                
                                                )
        

        self.Conv2d_21 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,stride=1),
                                        nn.ReLU(),

                                        )
        self.Conv2d_22 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,stride=1),
                                        nn.ReLU(),
                                            
                                            )
        self.Conv2d_23 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,stride=1),
                                        nn.ReLU(),

                                        )
        
        
        self.video_fine_conv = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1,stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1,stride=1),
                                        nn.ReLU(),
                                        )
        self.text_fine_shortcut_resizer = transpose2D([128,256])
        self.audio_fine_shortcut_resizer = transpose2D([128,256])
        self.video_fine_shortcut_resizer = transpose2D([128,256])

        self.text_coarse_shortcut_resizer = transpose2D([128,256])
        self.audio_coarse_shortcut_resizer = transpose2D([128,256])
        self.video_coarse_shortcut_resizer = transpose2D([128,256])

        self.addPara_audio = torch.nn.Parameter(torch.randn(32,32,16,32))
        self.addPara_video = torch.nn.Parameter(torch.randn(32,32,25,52))


        self.final_conv = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1,stride=1),
                                        nn.ReLU(),
        )
    
    def forward(self, text_fine, text_coarse, audio_fine, audio_coarse, video_fine, video_coarse):
        
        video_fine = self.video_fine_conv(video_fine)
        video_fine = video_fine.squeeze(1)
        
        # first shiffting
        text_merge = torch.cat([text_fine,text_coarse], dim=-1) 
        
        text_merge = self.Conv2d_11(text_merge)    

        audio_merge = torch.cat([audio_fine,audio_coarse], dim=-1)
        audio_merge = self.Conv2d_12(audio_merge)    

        video_merge = torch.cat([video_fine,video_coarse], dim=-1)
        video_merge = self.Conv2d_13(video_merge)  

        # shortcut
        text_shortcut_1 , text_shortcut_2=self.text_fine_shortcut_resizer(text_fine) ,self.text_coarse_shortcut_resizer( text_coarse)
        audio_shortcut_1 , audio_shortcut_2=self.audio_fine_shortcut_resizer(audio_fine) ,self.audio_coarse_shortcut_resizer(audio_coarse)
        video_shortcut_1 , video_shortcut_2=self.video_fine_shortcut_resizer(video_fine) ,self.video_coarse_shortcut_resizer(video_coarse)
    
        # first shortcut
        text_merge = text_merge + text_shortcut_1 
        audio_merge = audio_merge + audio_shortcut_1
        video_merge = video_merge + video_shortcut_1

        # second shiffting
        text_merge = self.Conv2d_21(text_merge) # torch.Size([32, 64, 24, 50])
        audio_merge = self.Conv2d_22(audio_merge) # torch.Size([32, 64, 8, 16])
        video_merge = self.Conv2d_23(video_merge) # torch.Size([32, 64, 6, 13])

        # second shortcut
        
        text_merge = text_merge + text_shortcut_2
        audio_merge = audio_merge + audio_shortcut_2
        video_merge = video_merge + video_shortcut_2

        # merge all the features
        features_merge = torch.cat([text_merge,audio_merge,video_merge], dim=-1) 
        
        # final shiffting
        features_merge = self.final_conv(features_merge) 
        
        return features_merge