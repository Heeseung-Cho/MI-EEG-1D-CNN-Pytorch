"""
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F

def SpatialDropout(input: torch.tensor, p: float = 0.5, inplace: bool = False):
    input = input.permute(0, 2, 1)   # convert to [batch, channels, time]
    input = F.dropout2d(input= input, p = p, inplace= inplace)
    input = input.permute(0, 2, 1)   # back to [batch, time, channels]
    return input

class HopefullNet(nn.Module):
    """
    Original HopeFullNet
    """
    def __init__(self, num_classes : int = 5,
                 kernal_size_0 : int = 20, 
                 kernel_size_1 : int = 6, 
                 drop_rate : float = 0.5):
        super(HopefullNet, self).__init__()       
        self.num_classes = num_classes 
        self.kernel_size_0 = kernal_size_0
        self.kernel_size_1 = kernel_size_1
        self.drop_rate = drop_rate
        
        # Layer1
        self.Layer1 = nn.Sequential(
            nn.ConstantPad1d((9, 10), 0.0),
            nn.Conv1d(in_channels= 2, out_channels= 32, kernel_size= self.kernel_size_0, stride = 1, padding= 0),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm1d(32)
        )
                
        # Layer2
        self.Layer2 = nn.Sequential(           
            nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= self.kernel_size_0, stride = 1, padding= 0),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm1d(32)  
        )                
        
        # Layer3
        self.Layer3 = nn.Sequential(           
            nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= self.kernel_size_1, stride = 1, padding= 0),
            nn.LeakyReLU(inplace = True)            
        )        
        
        #Layer4, Average Pooling to kernel_size = 2
        self.Layer4 = nn.AvgPool1d(kernel_size = 2)
        
        #Layer5
        self.Layer5 = nn.Sequential(           
            nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= self.kernel_size_1, stride = 1, padding= 0),
            nn.LeakyReLU(inplace = True),
        )        
        
        #FC Layer, Flatten and FC
        self.FC = nn.Sequential(
            nn.Linear(9696, 296),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(self.drop_rate),
            nn.Linear(296, 148),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(self.drop_rate),
            nn.Linear(148, 74),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(self.drop_rate),
            nn.Linear(74, self.num_classes)            
        )
    
    def forward(self, x : torch.tensor):
        out = self.Layer1(x)
        out = self.Layer2(out)
        out = SpatialDropout(out, p = self.drop_rate)
        out = self.Layer3(out)
        out = self.Layer4(out)
        out = self.Layer5(out)
        out = SpatialDropout(out, p = self.drop_rate)
        out = out.view(out.size(0), -1)
        out = self.FC(out)
        
        return out


if __name__ == '__main__':   
    input = torch.randn(2, 2, 640)
    model = HopefullNet()
    output = model(input)
    print(output)