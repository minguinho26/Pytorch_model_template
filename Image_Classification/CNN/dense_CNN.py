import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class transition_layer(nn.Module) :
    def __init__(self, channel) :
        super(transition_layer, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module('translation layer_' + str(channel) + '_1', nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0, bias = False))
        self.layer.add_module('translation layer_' + str(channel) + '_2', nn.AvgPool2d(3, stride=2))

    def forward(self, x) :
        return self.layer(x)

# 2개의 Conv2d로 구성된 residual block을 가지고 제작한 Residual. DenseNet에서 쓰임
# input_channel  : 입력값으로 들어오는 feature map의 channel
# output_channel : 출력값으로 나가는 feature map의 channel
class dense_residual_block(nn.Module) :
    def __init__(self, channel) :
        super(dense_residual_block, self).__init__()

        self.relu = nn.ReLU()

        self.block = nn.Sequential()
        self.block.add_module('resblock_BN_index_' + str(channel) + '_1', nn.BatchNorm2d(channel, affine=False))
        self.block.add_module('resblock_ReLU_index_' + str(channel) + '_1', nn.ReLU())
        self.block.add_module('resblock_conv2d_' + str(channel) + '_1', nn.Conv2d(channel, channel, kernel_size = 1, stride = 1, padding = 0, bias=False))
        self.block.add_module('resblock_BN_index_' + str(channel) + '_2', nn.BatchNorm2d(channel, affine=False))
        self.block.add_module('resblock_ReLU_index_' + str(channel) + '_2', nn.ReLU())
        self.block.add_module('resblock_conv2d_index_' + str(channel) + '_2', nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1, bias=False))
    
    def forward(self, x) :
        output = self.block(x)
        output += x
        output = self.relu(output)

        return output

# residual_block_num : Dense Block에 사용할 residual block의 개수
class dense_block(nn.Module) :
    def __init__(self, channel, residual_block_num) :
        super(dense_block, self).__init__()

        self.residual_block_list = []

        for i in range(residual_block_num) :
            self.residual_block_list.append(dense_residual_block(channel))
    
    def forward(self, x) :
        output_list = []

        output = self.residual_block_list[0](x)
        output += x 
        output_list.append(output)

        for i in range(1, len(self.residual_block_list)) :
            output = self.residual_block_list[i](x)
            # dense block의 dense conection을 위해 이전에 얻은 출력값과 현재 입력받은 입력값을 모두 출력값에 더해준다.
            output += x
            for pred_output in output_list : 
                output += pred_output

            output_list.append(output)
        return output

            
# input_channel : 입력받는 feature map의 채널.
# dense_block_first_channel : 맨처음 dense block이 사용할 channel. 다음 dense block으로 넘어갈 때마다 채널 개수가 2배씩 증가함
# dense_block_layer_list : 각 dense block에서 사용할 residual block의 개수. [6, 12, 24, 16] 등
# is_flatten : 마지막에 flatten() 연산을 할 것인가?
class dense_CNN(nn.Module) :
    def __init__(self, input_channel, dense_block_first_channel, dense_block_layer_list, is_flatten = False) :
        super(dense_CNN, self).__init__()

        self.dense_CNN = nn.Sequential(nn.Conv2d(input_channel, dense_block_first_channel, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                        )

        for i in range(len(dense_block_layer_list)) : 
            dense_block_channel = dense_block_first_channel * (2**i)
            self.dense_CNN.add_module('dense_block_' + str(dense_block_channel), dense_block(dense_block_channel, dense_block_layer_list[i]))
            # Dense Layer하고 1 x 1 conv로 채널수 조정 + avr pooling으로 이미지 크기 축소
            self.dense_CNN.add_module('translation layer_' + str(i), transition_layer(dense_block_channel))
        
        if is_flatten == True :
            self.dense_CNN.add_module('Flatten', nn.Flatten())

    def forward(self, x) :
        return self.dense_CNN(x)
    
    def model_summary(self, input_size_) :
        return summary(self.dense_CNN, input_size = input_size_)
        