import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# ResNet을 보면 Conv2d의 커널 개수가 증가할 때마다 feature map의 크기도 좌우상하 절반씩 줄어들고 extra zero entries가 padding된다. 이를 구현했다.
class channel_padding_and_pooling(nn.Module) :
    def __init__(self, input_feature, output_feature) :
        super(channel_padding_and_pooling, self).__init__()
        self.maxpooling = nn.MaxPool2d(3, stride=2)
        self.channel_diff_abs = abs(output_feature - input_feature)
    def forward(self, x) :
        x = self.maxpooling(x)

        return F.pad(input=x, pad=(0, 0, 0, 0, 0, self.channel_diff_abs, 0, 0), mode='constant', value=0) # (dim=0 앞뒤 pad), (dim=1 앞뒤 pad), (dim=2 앞뒤 pad), (dim=3 앞뒤 pad) 


# 2개의 Conv2d로 구성된 residual block을 가지고 제작한 Residual
# input_channel  : 입력값으로 들어오는 feature map의 channel
# output_channel : 출력값으로 나가는 feature map의 channel
class residual_block_small(nn.Module) :
    def __init__(self, channel) :
        super(residual_block_small, self).__init__()

        self.relu = nn.ReLU()

        self.block = nn.Sequential()
        self.block.add_module('resblock_conv2d_' + str(channel) + '_1', nn.Conv2d(channel, channel, kernel_size=3, stride = 1, padding = 1))
        self.block.add_module('resblock_BN_index_' + str(channel) + '_1', nn.BatchNorm2d(channel))
        self.block.add_module('resblock_ReLU_index_' + str(channel) + '_1', nn.ReLU())
        self.block.add_module('resblock_conv2d_index_' + str(channel) + '_2', nn.Conv2d(channel, channel, kernel_size=3, stride = 1, padding = 1))
        self.block.add_module('resblock_BN_index_' + str(channel) + '_2', nn.BatchNorm2d(channel))
    
    def forward(self, x) :
        output = self.block(x)
        output += x
        output = self.relu(output)

        return output

# 3개의 Conv2d로 구성된 residual block을 가지고 제작한 Residual 
# input_channel  : 입력값으로 들어오는 feature map의 channel
# output_channel : 출력값으로 나가는 feature map의 channel
class residual_block_big(nn.Module) :
    def __init__(self, channel) :
        super(residual_block_big, self).__init__()

        self.relu = nn.ReLU()

        self.block = nn.Sequential()
        self.block.add_module('resblock_conv2d_' + str(channel) + '_1', nn.Conv2d(channel, int(channel/4), kernel_size=1, stride = 1, padding = 0))
        self.block.add_module('resblock_BN_index_' + str(channel) + '_1', nn.BatchNorm2d(int(channel/4)))
        self.block.add_module('resblock_ReLU_index_' + str(channel) + '_1', nn.ReLU())
        self.block.add_module('resblock_conv2d_index_' + str(channel) + '_2', nn.Conv2d(int(channel/4), int(channel/4), kernel_size=3, stride = 1, padding = 1))
        self.block.add_module('resblock_BN_index_' + str(channel) + '_2', nn.BatchNorm2d(int(channel/4)))
        self.block.add_module('resblock_ReLU_index_' + str(channel) + '_2', nn.ReLU())
        self.block.add_module('resblock_conv2d_index_' + str(channel) + '_3', nn.Conv2d(int(channel/4), channel, kernel_size=1, stride = 1, padding = 0))
        self.block.add_module('resblock_BN_index_' + str(channel) + '_3', nn.BatchNorm2d(channel))

    def forward(self, x) :
        output = self.block(x)
        output += x
        output = self.relu(output)

        return output



# input_size : 입력되는 feature의 크기. 예를 들어 (3, 224, 224)
# kernal_list : CNN 제작에 사용할 Convolutional layer의 커널 개수들. Residual Block을 제작할 때 i번 째 kernal을 입력 channel, i + 1번 째 kernal을 출력 channel로 설정한 Residual Block을 제작한다. 
# num_classes : 분류할 클래스 개수
# Residual_Block_size : Residual block을 Conv2d 2개로 구성된 small을 사용할건지 3개로 구성된 big을 사용할건지 결정. 만약 big, small 외의 문자열을 입력했으면 small을 사용하게끔 설정됨
class residual_CNN(nn.Module) :
    def __init__(self, input_size, kernal_list, num_classes, Residual_Block_size = 'small', device = 'cuda') :
        super(residual_CNN, self).__init__()

        self.num_classes = num_classes
        self.device = device

        self.residual_CNN = nn.Sequential(nn.Conv2d(input_size[0], kernal_list[0], kernel_size=7, stride=2, padding=3),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                        )
        
        # Res Block 추가
        for i in range(len(kernal_list)) :
            
            if i != len(kernal_list) - 1 :

                if kernal_list[i] != kernal_list[i + 1] :
                    self.residual_CNN.add_module('padding_block_' + str(i), channel_padding_and_pooling(kernal_list[i], kernal_list[i + 1]))
                else :
                        self.residual_CNN.add_module('ResBlock_' + str(i), self.set_residual_block(Residual_Block_size, kernal_list[i]))
            else :
                self.residual_CNN.add_module('ResBlock_' + str(i), self.set_residual_block(Residual_Block_size, kernal_list[i]))

        self.residual_CNN.add_module('Flatten', nn.Flatten())
        self.residual_CNN.to(self.device)

        self.residual_CNN.eval()
        
        test_input = torch.unsqueeze(torch.ones(input_size), 0).to(self.device)
        test_output = self.residual_CNN(test_input)
        
        self.residual_CNN.train()
        
        
        self.header = nn.Sequential(
                nn.Linear(test_output.size()[1], self.num_classes),
                nn.Softmax(dim=1)
            ).to(self.device)

    def set_residual_block(self, Residual_Block_size, kernal_size) :
        if Residual_Block_size.lower() == 'big' :
            return residual_block_big(kernal_size)
        else :
            return residual_block_small(kernal_size)

    def forward(self, x) :
        output = self.residual_CNN(x)
        output = self.header(output)
        return output
    
    def model_summary(self, input_size_) :
        return summary(nn.Sequential(self.residual_CNN, self.header), input_size = input_size_)
