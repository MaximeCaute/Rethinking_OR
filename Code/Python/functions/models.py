import torch.nn as nn
import torch.nn.functional as F
import torch

from torch import optim

class Net_Final(nn.Module):
    def __init__(self, inp1, num_v, im_size, kernel):
        super().__init__()

        ins = 5
        self.cs = kernel
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp1, ins, kernel_size=self.cs),
            nn.BatchNorm2d(ins),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ins, 5, kernel_size=self.cs),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=self.cs),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(2*(im_size - 3*(self.cs-1))*(im_size - 3*(self.cs-1)), 264),
            nn.ReLU(),
        
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(264, num_v),
        )
        
        self.f_aux = nn.Sequential(
            nn.Linear(num_v, 2)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(7*num_v, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(500, num_v),
            nn.BatchNorm1d(num_v),
            nn.ReLU(),
        )
        

    def forward(self, x, v_x):
        num_v = v_x.shape[1] 
        
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x4 = x3.view(-1, 2*(x.shape[-1] - 3*(self.cs-1))*(x.shape[-1] - 3*(self.cs-1)))
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        
        x_aux = self.f_aux(x6)
        # add vehicles information
        x7 = torch.cat([v_x.transpose(2,1) ,x6.view(-1, 1,v_x.shape[1] )], dim=1).view(v_x.shape[0], -1)
        
        x8 = self.fc3(x7)
        x9 = self.fc4(x8)
        
        return x_aux, x9


class Net_Final_BIG(nn.Module):
    def __init__(self, inp1, num_v, im_size, kernel):
        super().__init__()

        ins = 10
        self.cs = kernel

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp1, ins, kernel_size=self.cs),
            nn.BatchNorm2d(ins),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ins, 30, kernel_size=self.cs),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(30, 50, kernel_size=self.cs),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(50 * (im_size - 3 * (self.cs - 1)) * (im_size - 3 * (self.cs - 1)), 500),
            nn.ReLU(),

        )

        self.fc2 = nn.Sequential(
            nn.Linear(500, num_v),
        )

        self.f_aux = nn.Sequential(
            nn.Linear(num_v, 2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(7 * num_v, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(500, num_v),
            nn.BatchNorm1d(num_v),
            nn.ReLU(),
        )

    def forward(self, x, v_x):
        num_v = v_x.shape[1]

        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = x3.view(-1, 50 * (x.shape[-1] - 3 * (self.cs - 1)) * (x.shape[-1] - 3 * (self.cs - 1)))
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)

        x_aux = self.f_aux(x6)
        #  add vehicles information
        x7 = torch.cat([v_x.transpose(2, 1), x6.view(-1, 1, v_x.shape[1])], dim=1).view(v_x.shape[0], -1)

        x8 = self.fc3(x7)
        x9 = self.fc4(x8)

        return x_aux, x9

class ImagesVal2(nn.Module):
    def __init__(self, inp1, num_v, im_size, kernel):
        super().__init__()

        ins = 5
        self.cs = kernel
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp1, ins, kernel_size=self.cs),
            nn.BatchNorm2d(ins),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ins, 5, kernel_size=self.cs),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=self.cs),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2*(im_size - 3*(self.cs-1))*(im_size - 3*(self.cs-1)), 264),
            nn.ReLU(),
        
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(264, num_v),
        )
        
        self.f_aux = nn.Sequential(
            nn.Linear(num_v, 2)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(7*num_v, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(300, num_v),
            nn.BatchNorm1d(num_v),
            nn.ReLU(),
        )
        
        

    def forward(self, x, v_x):
        num_v = v_x.shape[1]
        
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = x3.view(-1, 2*(x.shape[-1] - 3*(self.cs-1))*(x.shape[-1] - 3*(self.cs-1)))
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        
        x_aux = self.f_aux(x6)
        # add vehicles information
        x7 = torch.cat([v_x.transpose(2,1) ,x6.view(-1, 1,v_x.shape[1] )], dim=1).view(v_x.shape[0], -1)
        
        x8 = self.fc3(x7)
        x9 = self.fc4(x8)
        
        return x_aux, x9



class ImagesP2(nn.Module):
    def __init__(self, inp, num_v, im_size):
        super().__init__()

        ins = 10
        self.cs = 3 #set to 5 before
        self.im_size = im_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, ins, kernel_size=self.cs),
            nn.BatchNorm2d(ins),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ins, 5, kernel_size=self.cs),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=self.cs),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2*(im_size - 3*(self.cs-1))*(im_size - 3*(self.cs-1)), 500),
            nn.ReLU(),
        
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(500, num_v),
        )
        
        

    def forward(self, x):
                
        
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = x3.view(-1,2*(self.im_size - 3*(self.cs-1))*(self.im_size - 3*(self.cs-1)))
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        return x6


class ImagesP4(nn.Module):
    def __init__(self, inp, num_v, im_size):
        super().__init__()

        ins = 10
        self.cs = 3  # set to 5 before
        self.im_size = im_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, ins, kernel_size=self.cs),
            nn.BatchNorm2d(ins),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ins, 5, kernel_size=self.cs),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=self.cs),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2 * (im_size - 3 * (self.cs - 1)) * (im_size - 3 * (self.cs - 1)), 500),
            nn.ReLU(),

        )

        self.fc2 = nn.Sequential(
            nn.Linear(500, num_v),
        )

        self.f_aux = nn.Sequential(
            nn.Linear(2 * (im_size - 3 * (self.cs - 1)) * (im_size - 3 * (self.cs - 1)), 2)
        )

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = x3.view(-1, 2 * (self.im_size - 3 * (self.cs - 1)) * (self.im_size - 3 * (self.cs - 1)))
        x_aux = self.f_aux(x4)
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        return x_aux, x6
    

class ImagesP3(nn.Module):
    def __init__(self, inp1, num_v, im_size, kernel):
        super().__init__()

        ins = 5
        self.cs = kernel
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp1, ins, kernel_size=self.cs),
            nn.BatchNorm2d(ins),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ins, 5, kernel_size=self.cs),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=self.cs),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(2*(im_size - 3*(self.cs-1))*(im_size - 3*(self.cs-1)), 264),
            nn.ReLU(),
        
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(264, num_v),
        )
        
        self.f_aux = nn.Sequential(
            nn.Linear(num_v, 2)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(7*num_v, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(300, num_v),
            nn.BatchNorm1d(num_v),
            nn.ReLU(),
        )
        

    def forward(self, x, v_x):
        num_v = v_x.shape[1] 
        
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x4 = x3.view(-1, 2*(x.shape[-1] - 3*(self.cs-1))*(x.shape[-1] - 3*(self.cs-1)))
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        
        x_aux = self.f_aux(x6)
        # add vehicles information
        x7 = torch.cat([v_x.transpose(2,1) ,x6.view(-1, 1,v_x.shape[1] )], dim=1).view(v_x.shape[0], -1)
        
        x8 = self.fc3(x7)
        x9 = self.fc4(x8)
        
        return x_aux, x9