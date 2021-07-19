import torch
import torch.nn as nn
import torch.nn.functional as F


class testNet_2class(nn.Module):

    """
    A simple Model for sanity checks.
    """

    def __init__(self, num_pathologies, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[1], 4, 12, stride=6)
        self.pool1 = nn.MaxPool2d(2)
        
        out_size = int(((input_size[0]-12)/6)+1)
        after_pool = int(((out_size-2)/2)+1)
        self.fc1_size = (after_pool**2)*4
        #print(self.fc1_size)
        
        self.fc1 = nn.Linear(self.fc1_size, 32)
        self.fc2 = nn.Linear(32, num_pathologies+1)
        

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, self.fc1_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, num_pathologies, two_class=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(8, 10, 3, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9610, 100)
        if not two_class:
            self.fc2 = nn.Linear(100, 3 * num_pathologies + 2)
        else:
            self.fc2 = nn.Linear(100, num_pathologies + 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # print(x.size())

        x = x.view(-1, 9610)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
def calc_out_shape(size, kernel, padding, stride):
    return int(((size-kernel+2*padding)/stride)+1)
    
class ResBlockDrop(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.drop1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        return x
    
class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(num_channels)
        self.drop1 = nn.Dropout(p=0.25)
        self.conv_out = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        x_in = x
        x = self.relu(self.conv_in(x))
        x = self.batch1(x)
        x = self.drop1(x)
        x = self.relu(self.conv_out(x))
        x = self.batch2(x)
        return x + x_in

class ResNetSimple(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[1], 128, 7, stride=2)
        out_conv1 = calc_out_shape(input_size[0], 7, 0, 2.)
        print(out_conv1)
        
        self.pool1 = nn.MaxPool2d(3, stride=2)
        out_pool1 = calc_out_shape(out_conv1, 3, 0, 2.)
        print(out_pool1)
        
        self.conv2 = nn.Conv2d(128, 512, 5, stride=2)
        out_conv2 = calc_out_shape(out_pool1, 5, 0, 2.)
        print(out_conv2)
        
        self.pool2 = nn.MaxPool2d(3, stride=2)
        out_pool2 = calc_out_shape(out_conv2, 3, 0, 2.)
        print(out_pool2)
        
        #self.conv3 = nn.Conv2d(128, 512, 2, stride=1)
        #out_conv3 = calc_out_shape(out_pool2, 2, 0, 1.)
        #print(out_conv3)
        
        #self.pool3 = nn.MaxPool2d(2)
        #out_pool3 = calc_out_shape(out_conv3, 2, 0, 2.)
        #print(out_pool3)
        
        #self.conv4 = nn.Conv2d(48, 64, 2, stride=1)
        #out_conv4 = calc_out_shape(out_pool3, 2, 0, 1.)
        #print(out_conv4)
        
        #self.pool4 = nn.MaxPool2d(2)
        #self.out = (calc_out_shape(out_conv4, 2, 0, 2.)**2)*64
        #print(self.out)
        
        self.resBlock1 = ResBlock(512)
        self.resBlock2 = ResBlock(512)
        #self.resBlock3 = ResBlock(32)
        
        self.pool4 = nn.MaxPool2d(7)
        self.out = (calc_out_shape(out_pool2, 7, 0, 7.)**2)*512
        print(self.out)
        
        self.fc1 = nn.Linear(self.out, 2048)
        self.fc2 = nn.Linear(2048, num_pathologies+1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #x = F.relu(self.conv3(x))
        #x = self.pool3(x)
        #x = F.relu(self.conv4(x))
        #x = self.pool4(x)
        
        y1 = self.resBlock1(x)
        res_x1 = x + y1
        
        y2 = self.resBlock2(res_x1)
        res_out = res_x1 + y2
        
        #y3 = self.resBlock3(res_x2)
        #res_out = res_x1 +y3
        
        res_out = self.pool4(res_out)
        res_out = res_out.view(-1, self.out)
        res_out = F.relu(self.fc1(res_out))
        res_out = self.fc2(res_out)
        #res_out = F.relu(self.fc2(res_out))
        #res_out = self.fc3(res_out)
        return res_out
    
    
class ResNetSimple2(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[1], 128, 7, stride=2)
        out_conv1 = calc_out_shape(input_size[0], 7, 0, 2.)
        print(out_conv1)
        
        self.pool1 = nn.MaxPool2d(3, stride=2)
        out_pool1 = calc_out_shape(out_conv1, 3, 0, 2.)
        print(out_pool1)
        
        self.resBlock1 = ResBlock(128)
        
        self.conv2 = nn.Conv2d(128, 512, 5, stride=2)
        out_conv2 = calc_out_shape(out_pool1, 5, 0, 2.)
        print(out_conv2)
        
        self.pool2 = nn.MaxPool2d(3, stride=2)
        out_pool2 = calc_out_shape(out_conv2, 3, 0, 2.)
        print(out_pool2)
        
        self.resBlock2 = ResBlock(512)
        
        #self.conv3 = nn.Conv2d(128, 512, 2, stride=1)
        #out_conv3 = calc_out_shape(out_pool2, 2, 0, 1.)
        #print(out_conv3)
        
        #self.pool3 = nn.MaxPool2d(2)
        #out_pool3 = calc_out_shape(out_conv3, 2, 0, 2.)
        #print(out_pool3)
        
        #self.conv4 = nn.Conv2d(48, 64, 2, stride=1)
        #out_conv4 = calc_out_shape(out_pool3, 2, 0, 1.)
        #print(out_conv4)
        
        #self.pool4 = nn.MaxPool2d(2)
        #self.out = (calc_out_shape(out_conv4, 2, 0, 2.)**2)*64
        #print(self.out)
        
        
        #self.resBlock3 = ResBlock(512)
        #self.resBlock3 = ResBlock(32)
        
        self.pool4 = nn.AvgPool2d(7)
        self.out = (calc_out_shape(out_pool2, 7, 0, 7.)**2)*512
        print(self.out)
        
        self.fc1 = nn.Linear(self.out, 2048)
        self.fc2 = nn.Linear(2048, num_pathologies+1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        y1 = self.resBlock1(x)
        res_x1 = x + y1
        
        res_x1 = F.relu(self.conv2(res_x1))
        res_x1 = self.pool2(res_x1)
        #
        #x = F.relu(self.conv4(x))
        #x = self.pool4(x)
        
        
        
        y2 = self.resBlock2(res_x1)
        res_out = res_x1 + y2
        
        #res_x2 = F.relu(self.conv3(res_x2))
        #res_x2 = self.pool3(res_x2)
        
        
        #y3 = self.resBlock3(res_x2)
        #res_out = res_x2 +y3
        
        res_out = self.pool4(res_out)
        res_out = res_out.view(-1, self.out)
        res_out = F.relu(self.fc1(res_out))
        res_out = self.fc2(res_out)
        #res_out = F.relu(self.fc2(res_out))
        #res_out = self.fc3(res_out)
        return res_out
    
class ResNetSimple3(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_size[1], 128, 7, stride=2)
        out_conv1 = calc_out_shape(input_size[0], 7, 0, 2.)
        #print(out_conv1)
        
        self.pool1 = nn.MaxPool2d(3, stride=2)
        out_pool1 = calc_out_shape(out_conv1, 3, 0, 2.)
        #print(out_pool1)
        self.batch1 = nn.BatchNorm2d(128)
        self.resBlock1 = ResBlock(128)
        
        self.conv2 = nn.Conv2d(128, 512, 5, stride=2)
        out_conv2 = calc_out_shape(out_pool1, 5, 0, 2.)
        #print(out_conv2)
        
        self.pool2 = nn.MaxPool2d(3, stride=2)
        out_pool2 = calc_out_shape(out_conv2, 3, 0, 2.)
        #print(out_pool2)
        
        self.batch2 = nn.BatchNorm2d(512)
        
        self.resBlock2 = ResBlock(512)
        
        self.conv3 = nn.Conv2d(512, 1028, 3, stride=2)
        out_conv3 = calc_out_shape(out_pool2, 3, 0, 2.)
        #print(out_conv3)
        
        #self.pool3 = nn.MaxPool2d(2)
        #out_pool3 = calc_out_shape(out_conv3, 2, 0, 2.)
        #print(out_pool3)
        
        #self.conv4 = nn.Conv2d(48, 64, 2, stride=1)
        #out_conv4 = calc_out_shape(out_pool3, 2, 0, 1.)
        #print(out_conv4)
        
        #self.pool4 = nn.MaxPool2d(2)
        #self.out = (calc_out_shape(out_conv4, 2, 0, 2.)**2)*64
        #print(self.out)
        
        
        #self.resBlock3 = ResBlock(512)
        self.resBlock3 = ResBlock(1028)
        
        self.pool3 = nn.MaxPool2d(6)
        self.out = (calc_out_shape(out_conv3, 6, 0, 6.)**2)*1028
        #print(self.out)
        
        self.fc1 = nn.Linear(self.out, 1028)
        self.fc2 = nn.Linear(1028, num_pathologies+1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batch1(x)
        
        y1 = self.resBlock1(x)
        res_x1 = x + y1
        
        res_x1 = self.relu(self.conv2(res_x1))
        res_x1 = self.pool2(res_x1)
        res_x1 = self.batch2(res_x1)
        #
        #x = F.relu(self.conv4(x))
        #x = self.pool4(x)
        
        
        
        y2 = self.resBlock2(res_x1)
        res_x2 = res_x1 + y2
        
        res_x2 = self.relu(self.conv3(res_x2))
        
        #res_x2 = F.relu(self.conv3(res_x2))
        #res_x2 = self.pool3(res_x2)
        
        
        y3 = self.resBlock3(res_x2)
        res_out = res_x2 +y3
        
        res_out = self.pool3(res_out)
        res_out = res_out.view(-1, self.out)
        res_out = self.relu(self.fc1(res_out))
        res_out = self.fc2(res_out)
        #res_out = F.relu(self.fc2(res_out))
        #res_out = self.fc3(res_out)
        return res_out
    
    
class ResNetSimple4(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_size[1], 128, 7)
        out_conv1 = calc_out_shape(input_size[0], 7, 0, 1.)
        #print(out_conv1)
        
        self.pool1 = nn.MaxPool2d(6, stride=3)
        out_pool1 = calc_out_shape(out_conv1, 6, 0, 3.)
        #print(out_pool1)
        self.batch1 = nn.BatchNorm2d(128)
        self.resBlock1 = ResBlock(128)
        
        self.conv2 = nn.Conv2d(128, 512, 3)
        out_conv2 = calc_out_shape(out_pool1, 3, 0, 1.)
        #print(out_conv2)
        
        self.pool2 = nn.MaxPool2d(6, stride=3)
        out_pool2 = calc_out_shape(out_conv2, 6, 0, 3.)
        #print(out_pool2)
        
        self.batch2 = nn.BatchNorm2d(512)
        
        self.resBlock2 = ResBlock(512)
        
        self.conv3 = nn.Conv2d(512, 1028, 3)
        out_conv3 = calc_out_shape(out_pool2, 3, 0, 1.)
        
        self.pool3 = nn.MaxPool2d(4, stride=3)
        out_pool3 = calc_out_shape(out_conv3, 4, 0, 3.)
        self.batch3 = nn.BatchNorm2d(1028)
        #print(out_pool3)
        
        self.resBlock3 = ResBlock(1028)
        
        self.pool4 = nn.MaxPool2d(7)
        self.out = (calc_out_shape(out_pool3, 7, 0, 7.)**2)*1028
        #print(self.out)
        
        self.fc1 = nn.Linear(self.out, 1028)
        self.fc2 = nn.Linear(1028, num_pathologies+1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batch1(x)
        
        x = self.resBlock1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batch2(x)
        
        
        x = self.resBlock2(x)
        
        x = self.batch3(self.pool3(self.relu(self.conv3(x))))

        x = self.resBlock3(x)
        
        x = self.pool4(x)
        x = x.view(-1, self.out)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ResNetSimple5(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_size[1], 64, 12)
        out_conv1 = calc_out_shape(input_size[0], 12, 0, 1.)
        #print(out_conv1)
        
        self.pool1 = nn.MaxPool2d(2)
        out_pool1 = calc_out_shape(out_conv1, 2, 0, 2.)
        #print(out_pool1)
        self.batch1 = nn.BatchNorm2d(64)
        self.resBlock1 = ResBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, 6)
        out_conv2 = calc_out_shape(out_pool1, 6, 0, 1.)
        #print(out_conv2)
        
        self.pool2 = nn.MaxPool2d(2)
        out_pool2 = calc_out_shape(out_conv2, 2, 0, 2.)
        #print(out_pool2)
        
        self.batch2 = nn.BatchNorm2d(128)
        
        self.resBlock2 = ResBlock(128)
        
        self.conv3 = nn.Conv2d(128, 512, 4)
        out_conv3 = calc_out_shape(out_pool2, 4, 0, 1.)
        
        self.pool3 = nn.MaxPool2d(2)
        out_pool3 = calc_out_shape(out_conv3, 2, 0, 2.)
        self.batch3 = nn.BatchNorm2d(512)
        #print(out_pool3)
        
        self.resBlock3 = ResBlock(512)
        
        self.conv4 = nn.Conv2d(512, 1028, 3)
        out_conv4 = calc_out_shape(out_pool3, 3, 0, 1.)
        self.pool4 = nn.MaxPool2d(2)
        out_pool4 = calc_out_shape(out_conv4, 2, 0, 2.)
        self.batch4 = nn.BatchNorm2d(1028)
        print(out_pool4)
        
        self.resBlock4 = ResBlock(1028)
        
        self.pool5 = nn.MaxPool2d(12)
        self.out = (calc_out_shape(out_pool4, 12, 0, 12.)**2)*1028
        #print(self.out)
        
        self.fc1 = nn.Linear(self.out, 1028)
        self.fc2 = nn.Linear(1028, num_pathologies+1)
        
        #self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batch1(x)
        
        x = self.resBlock1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batch2(x)
        
        
        x = self.resBlock2(x)
        
        x = self.batch3(self.pool3(self.relu(self.conv3(x))))

        x = self.resBlock3(x)
        
        x = self.batch4(self.pool4(self.relu(self.conv4(x))))
        x = self.resBlock4(x)
        x = self.pool5(x)
        x = x.view(-1, self.out)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ResNetSimple6(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(input_size[1], 64, 7)
        self.conv1_2 = nn.Conv2d(64, 128, 5)
        out_conv1 = calc_out_shape(calc_out_shape(input_size[0], 7, 0, 1.), 5, 0, 1.)
        self.pool1 = nn.MaxPool2d(2)
        #print(out_conv1)
        
        out_pool1 = calc_out_shape(out_conv1, 2, 0, 2.)
        #print(out_pool1)
        self.batch1 = nn.BatchNorm2d(128)
        self.resBlock1_1 = ResBlock(128)
        #self.resBlock1_2 = ResBlock(128)
        
        self.conv2 = nn.Conv2d(128, 512, 1)
        self.batch2 = nn.BatchNorm2d(512)
        #print(out_conv2)
        self.resBlock2_1 = ResBlock(512)
        #self.resBlock2_2 = ResBlock(512)
        self.pool2 = nn.MaxPool2d(2)
        out_pool2 = calc_out_shape(out_pool1, 2, 0, 2.)
        #print(out_pool2)
        
        
        self.conv3 = nn.Conv2d(512, 1028, 1)
        self.batch3 = nn.BatchNorm2d(1028)
        #self.resBlock3_1 = ResBlock(1028)
        self.resBlock3_1 = ResBlock(1028)
        
        self.pool3 = nn.MaxPool2d(2)
        out_pool3 = calc_out_shape(out_pool2, 2, 0, 2.)
        #print(out_pool3)
        
        #self.conv4 = nn.Conv2d(1028, 2048, 1)
        #self.resBlock4_1 = ResBlock(2048)
        self.resBlock4 = ResBlock(1028)
        self.pool4 = nn.MaxPool2d(2)
        out_pool4 = calc_out_shape(out_pool3, 2, 0, 2.)
        print(out_pool4)
        self.final_pool = nn.MaxPool2d(out_pool4)
        
        self.out = 1028
        print(self.out)
        """
        self.conv4 = nn.Conv2d(512, 1028, 3)
        out_conv4 = calc_out_shape(out_pool3, 3, 0, 1.)
        self.pool4 = nn.MaxPool2d(2)
        
        self.batch4 = nn.BatchNorm2d(1028)
        print(out_pool4)
        
        self.resBlock4 = ResBlock(1028)
        
        self.pool5 = nn.MaxPool2d(12)
        self.out = (calc_out_shape(out_pool4, 12, 0, 12.)**2)*1028
        #print(self.out)
        """
        
        self.fc1 = nn.Linear(self.out, 1028)
        self.fc2 = nn.Linear(1028, num_pathologies+1)
        
        #self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.batch1(self.pool1(
                        self.relu(self.conv1_2(
                            self.relu(self.conv1_1(x))))))
        
        x = self.pool2(self.resBlock1_1(x))
        
        x = self.batch2(self.relu(self.conv2(x)))
        x = self.pool3(self.resBlock2_1(x))
        
        x = self.batch3(self.relu(self.conv3(x)))
        x = self.pool4(self.resBlock3_1(x))
        
        x = self.final_pool(self.resBlock4(x))
        
        x = x.view(-1, self.out)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_pathologies, input_size, two_class=True):
        super().__init__()
        self.conv1_1 = nn.Conv2d(input_size[1], 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2)
        
        out1 = calc_out_shape(calc_out_shape(calc_out_shape(input_size[0], 3, 0, 1.), 3, 0, 1,), 2, 0, 2.)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2)
        
        out2 = calc_out_shape(calc_out_shape(calc_out_shape(out1, 3, 0, 1.), 3, 0, 1.), 2, 0, 2.)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.pool3 = nn.MaxPool2d(2)
        
        out3 = calc_out_shape(calc_out_shape(calc_out_shape(calc_out_shape(out2, 3, 0, 1.), 3, 0, 1.), 3, 0, 1.), 2, 0, 2.)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.pool4 = nn.MaxPool2d(2)
        
        out4 = calc_out_shape(calc_out_shape(calc_out_shape(calc_out_shape(out3, 3, 0, 1.), 3, 0, 1.), 3, 0, 1), 2, 0, 2.)
        
        self.conv5_1 = nn.Conv2d(256, 512, 3)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2)
        
        self.out = calc_out_shape(calc_out_shape(calc_out_shape(calc_out_shape(out4, 3, 0, 1.), 3, 0, 1.), 3, 0, 1.), 2, 0, 2.)**2 * 512
        
        self.fc1 = nn.Linear(self.out, 2048)
        self.fc2 = nn.Linear(2048, 1000) 
        self.fc3 = nn.Linear(1000, num_pathologies+1)
        
        print(self.out)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.pool1(
            self.relu(self.conv1_2(
                self.relu(self.conv1_1(x)))))
        
        x = self.pool2(
            self.relu(self.conv2_2(
                self.relu(self.conv2_1(x)))))
        
        x = self.pool3(
            self.relu(self.conv3_3(
                self.relu(self.conv3_2(
                    self.relu(self.conv3_1(x)))))))
        
        x = self.pool4(
            self.relu(self.conv4_3(
                self.relu(self.conv4_2(
                    self.relu(self.conv4_1(x)))))))
        
        x = self.pool5(
            self.relu(self.conv5_3(
                self.relu(self.conv5_2(self.relu(self.conv5_2(x)))))))
        
        x = x.view(-1, self.out)
        
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        
        return x
        
        
    