import torch.nn as nn
from torchsummary import summary

# input_channel : 입력되는 feature의 채널 수. 3채널 이미지의 경우 3.
# kernal_list : [64, 64, 'p', 128, 128]의 양식을 사용. p는 pooling을 뜻함
# activation_func : Conv2d 이후 사용할 활성화 함수. 맨 마지막 Conv2d에는 사용하지 않음. 
# is_flatten : Conv2d 연산을 끝내고 맨 마지막에 일차원 벡터로 만들지 말지 결정
# batch_normalization_use : BN을 사용할지 말지 결정
class normal_CNN(nn.Module) :
    def __init__(self, input_channel, kernal_list, activation_func = 'relu',  is_flatten = False, batch_normalization_use = True) :
        super(normal_CNN, self).__init__()

        self.CNN = nn.Sequential()

        for i in range(len(kernal_list)) :
            # pooling도 고려

            # Convolutional Layer 추가

            # 만약 pooling Layer를 추가할 경우
            if kernal_list[i] == 'p' :
                self.CNN.add_module("Pooling_" + str(i), nn.MaxPool2d(3, stride=2))
                kernal_list[i] = kernal_list[i - 1] # [64, 'p', 128] -> [64, 64 ,128]로 만들어서 nn.Conv2d(kernal_list[i - 1], kernal_list[i]... 명령어를 수행할 때 에러가 나지 않게 만든다.
            else :
                if i == 0 :
                    self.CNN.add_module("Conv2d_" + str(i), nn.Conv2d(input_channel, kernal_list[i], kernel_size=3, stride = 1, padding = 1, bias=False))
                else :
                    self.CNN.add_module("Conv2d_" + str(i), nn.Conv2d(kernal_list[i - 1], kernal_list[i], kernel_size=3, stride = 1, padding = 1, bias=False))
                # BN 사용 여부
                if batch_normalization_use == True :
                    self.CNN.add_module("BN_" + str(i), nn.BatchNorm2d(kernal_list[i], affine=False))

                if i != len(kernal_list) - 1 :
                    if activation_func.lower() == 'relu' : 
                        self.CNN.add_module('ReLU_' + str(i), nn.ReLU())
                    elif activation_func.lower() == 'leaky_relu' : 
                        self.CNN.add_module('LeakyReLU_' + str(i), nn.LeakyReLU())

        if is_flatten == True :
            self.CNN.add_module('Flatten', nn.Flatten())

    def forward(self, x) :
        return self.CNN(x)
    
    def model_summary(self, input_size_) :
        return summary(self.CNN, input_size=input_size_)