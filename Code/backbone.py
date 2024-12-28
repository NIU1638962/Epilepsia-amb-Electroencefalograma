import torch
import torch.nn as nn


class InputLevelFusion(nn.Module):
    def __init__(self, num_channels=21):
        super(InputLevelFusion, self).__init__()
        
        # Fusion de canales
        self.feature_fusion = nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=3, padding=1)
        
        # Extraccion de features con CNN
        self.channel_projection = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernelsize=4)
            )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128, out_features=2)
    
    def forward(self, x):

        x = self.feature_fusion(x)  
        x = self.channel_projection(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x




class FeatureLevelFusion(nn.Module):
    def __init__(self, num_channels=21):
        super(FeatureLevelFusion, self).__init__()
        
        # Fusión de canales usando CNN
        self.channel_feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernelsize=4)
            ) for _ in range(num_channels)
        ])
        
        # Proyección de características fusionadas
        self.feature_fusion = nn.Conv1d(in_channels=num_channels * 64, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=128, out_features=2)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        channel_features = []
        for i in range(x.shape[1]):
            channel = x[:, i, :].unsqueeze(1)
            channel_feature = self.channel_feature_extractors[i](channel)
            channel_features.append(channel_feature)
        
        # Concatenar características de todos los canales
        x = torch.cat(channel_features, dim=1)
        
        # Fusionar características
        x = self.feature_fusion(x) 
        x = self.flatten(x)
        x = self.fc(x)
        
        return x