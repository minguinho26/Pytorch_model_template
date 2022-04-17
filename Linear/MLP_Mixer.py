import torch
import torch.nn as nn
import torch.nn.functional as F

class Per_patch_Fully_connected(nn.Module) :
    def __init__(self, input_size, patch_size, C) :
        super(Per_patch_Fully_connected, self).__init__()

        # 이미지 넣기 전에 특정 규격으로 resize 해줄 필요가 있어보인다
        self.S = int((input_size[1] * input_size[2]) / (patch_size ** 2))
        self.projection_layer = nn.Linear(input_size[0] * patch_size * patch_size,  C, bias = False) # desired hidden dimension C로 embedding함
        # projection_layer의 in_feature가 input_size[0] * patch_size * patch_size인 이유 : 이미지의 channel * H * W를 S로 나누니까 [channel * patch size * patch size]가 나와서

    def forward(self, x) :
        # [3, H, W] 크기의 이미지를 [S, ]
        x = torch.reshape(x, (self.S, -1)) # x를 (S, channel*patch_size^2) 형태로 만들어준다

        return self.projection_layer(x) # [S, C] 크기의 vector 반환(C차원의 값을 가지고 있는 S개의 token)


# N층의 mixer layer들은 각각 token-mixing MLP, channel-mixing-MLP로 구성되어 있으며 각 MLP는 [linear -> GELU -> linear]로 구성되어 있음
# Figure 1에 나온대로 만들면 됨

# token-mixing MLP
# [S x C] 크기의 embedded patch 그룹을 transpose 후 처리. 즉, [C x S] 벡터를 처리하는데 S개의 token이 있다고 했으니 '같은 채널'에 있던 각 영역의 값들을 계산에 사용하는거임. 
# 이를 논문에서는 'allow communication between different spatial locations'라고 표시
# input_size : [S, C] 크기의 벡터. 예를 들어 (196, 16)과 같은 튜플을 넣어주면 된다. 리스트도 가능. 
class token_mixing_MLP(nn.Module) : 
    def __init__(self, input_size) : 
        super(token_mixing_MLP, self).__init__()

        self.Layer_Norm = nn.LayerNorm([input_size[0], input_size[1]])
        self.MLP = nn.Sequential(
            nn.Linear(input_size[0], input_size[0], bias = False),
            nn.GELU(),
            nn.Linear(input_size[0], input_size[0], bias = False)
        )

    def forward(self, x) :
        # layer_norm + transpose
        output = self.Layer_Norm(x)
        output = torch.transpose(output, 0, 1) # [C x S] 형태가 됨
        output = self.MLP(output)

        # [S x C] 형태로 transpose + skip connection
        output = torch.transpose(output, 0, 1)

        return output + x

# channel-mixing_MLP
# 각 patch에 있던 값들을 계산에 사용. 위치별로 잘라서 계산하니까 해당 위치에 존재하는 모든 채널에서 얻은 값들이 연산에 사용된다. -> allow communication between different channels
class channel_mixing_MLP(nn.Module) :
    def __init__(self, input_size) : # 
        super(channel_mixing_MLP, self).__init__()

        self.Layer_Norm = nn.LayerNorm([input_size[0], input_size[1]])

        self.MLP = nn.Sequential(
            nn.Linear(input_size[1], input_size[1], bias = False),
            nn.GELU(),
            nn.Linear(input_size[1], input_size[1], bias = False)
        )
    
    def forward(self, x) :
        output = self.Layer_Norm(x)
        output = self.MLP(output)

        return output + x

# 앞서 구현한 token_mixing_MLP, channel_mixing_MLP을 합체
# input_size : [S, C] 크기의 벡터
class Mixer_Layer(nn.Module) :
    def __init__(self, input_size) : # 
        super(Mixer_Layer, self).__init__()

        self.mixer_layer = nn.Sequential(
            token_mixing_MLP(input_size),
            channel_mixing_MLP(input_size)
        )
    def forward(self, x) :
        return self.mixer_layer(x)


# MLP-Mixer
# Per_patch_Fully_connected, token_mixing_MLP, channel_mixing_MLP로 구성됨
# input_size : 입력할 이미지 사이즈. (C, H, W) 양식이다. 예를 들면 (3, 224, 224)
# patch_size : 모델이 사용할 patch의 사이즈. 예를 들어 16
# C : desired hidden dimension. 예를 들어 16
# N : Mixer Layer의 개수
# classes_num : 분류해야하는 클래스의 개수
class MLP_Mixer(nn.Module) :
    def __init__(self, input_size, patch_size, C, N, classes_num) : 
        super(MLP_Mixer, self).__init__()

        S = int((input_size[1] * input_size[2]) / (patch_size ** 2)) # embedding으로 얻은 token의 개수

        self.mlp_mixer = nn.Sequential(
            Per_patch_Fully_connected(input_size, patch_size, C)
        )
        for i in range(N) : # Mixer Layer를 N번 쌓아준다
            self.mlp_mixer.add_module("Mixer_Layer_" + str(i), Mixer_Layer((S, C)))
        
        # Glboal Average Pooling
        # Appendix E에 pseudo code가 있길래 그거 보고 제작
        # LayerNorm 하고 token별로 평균을 구한다
        self.global_average_Pooling_1 = nn.LayerNorm([S, C])

        self.head = nn.Sequential(
            nn.Linear(S, classes_num, bias = False),
            nn.Softmax(dim=0)
        )

    def forward(self, x) :
        output = self.mlp_mixer(x)
        output = self.global_average_Pooling_1(output)
        output = torch.mean(output, 1)

        return self.head(output)


# 사용 예시============================================================================
# image_size = (3, 224, 224)
# input = torch.ones(image_size)
# mixer = MLP_Mixer(input_size=image_size, patch_size=32, C=512, N=8, classes_num=1000)
# output = mixer(input)
# 사용 예시============================================================================