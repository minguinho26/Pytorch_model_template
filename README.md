# Pytorch_model_template

제작자 : 김민규 <br>
리포지토리 생성일 : 22.4.16

## 설명
파이토치를 이용해 제작한 네트워크(MLP, CNN 등)들을 쉽게 사용하기 위해 제작하는 템플릿. <br>
22.4.16 기준으로 3가지 유형(Linear, CNN, residual CNN)의 네트워크를 구현했습니다. 

1. Linear : 선형 연산을 수행하는 nn.Linear 레이어를 가지고 구현했습니다. <br> 사용 예)
    ~~~python
    from Linear.normal_MLP import *

    # 기본적으로 연산 도중에 사용하는 활성화 함수는 ReLU, 연산의 마지막에 사용하는 활성화 함수는 Sigmoid 함수로 설정했습니다. 그리고 Batch Normalization도 사용하게 설정했습니다. 이 설정들은 초기화 할 때 입력하는 값을 통해 개인적으로 변경하실 수 있습니다. 
    model = normal_MLP(input_size = 64, neural_list = [64, 64, 128, 128],  mid_activation_func = 'leaky_relu',  last_activateion_func = 'softmax', batch_normalization_use = False)
    ~~~

2. CNN : Covolution 연산을 수행하는 네트워크입니다. <br> 사용 예)
    ~~~python
    # 일반적인 CNN을 사용하는 경우
    from CNN.normal_CNN import *

    # input_channel : 입력하는 값의 채널. 3채널 이미지의 경우 3입니다. 
    # kernal_list : Convolutional layer들이 사용할 커널의 개수. p는 pooling을 나타냅니다. 
    # is_flatten : 연산의 마지막에 1차원 벡터로 만들어주는 nn.Flatten()을 사용할 것인지 말지 결정합니다. 
    # batch_normalization_use : 연산 중간중간에 Batch Normalizatoin을 사용할지 말지 결정합니다.
    # 기본적으로 연산 도중에 사용할 활성화 함수로 ReLU를 사용하고 연산의 마지막에 flatten 연산을 수행하지 않게 설정했습니다. 이 역시 초기화시 입력하는 값을 통해 수정할 수 있습니다. 그리고 Batch Normalization도 기본적으로 사용하게끔 설정되어 있으나 역시 수정 가능합니다. 
    model = normal_CNN(input_channel = input_channel, kernal_list = [64, 64, 'p', 128, 128], activation_func = 'leaky_relu',  is_flatten = True, batch_normalization_use = False)
    ~~~

    ~~~python
    # Residual CNN, 그러니까 ResNet 계열의 CNN을 사용하는 경우
    from CNN.residual_CNN import *

    # input_channel : 입력하는 값의 채널. 3채널 이미지의 경우 3입니다. 
    # kernal_list : Convolutional layer들이 사용할 커널의 개수. residual_CNN은 커널이 변경될 때마다 pooling을 수행합니다.  
    # Residual_Block_size : small이면 2개의 Convolutional layer로 구성된 residual block 사용, big이면 3개의 Convolutional layer로 구성된 residual block 사용
    # is_flatten : 연산의 마지막에 1차원 벡터로 만들어주는 nn.Flatten()을 사용할 것인지 말지 결정합니다. 

    # ResNet 논문에 따르면 50층 이상의 큰 네트워크를 설계할 때는 3개의 Convolutional layer로 구성된 residual block을 쓰고 그보다 작은 네트워크를 설계할 때는 2개의 Convolutional layer로 구성된 residual block을 사용합니다. 
    model = residual_CNN(input_channel = 3, kernal_list = [64, 64, 128, 128], Residual_Block_size = 'big', is_flatten = True)
    ~~~

    ~~~python
    # Dense CNN, DenseNet을 구성하는 Dense block으로 구성된 CNN
    from CNN.dense_CNN import *

    # input_channel : 입력하는 값의 채널. 3채널 이미지의 경우 3입니다.
    # dense_block_first_channel. 맨처음 Dense Block이 사용하는 커널의 개수. DenseNet은 다음 Dense block으로 넘어갈 때마다 채널의 크기를 2배씩 늘립니다. 그래서 맨처음 channel, 즉 kernal의 값만 받습니다.
    # dense_block_layer_list. 각 Dense block에서 쓰이는 residual block의 개수를 나타냅니다. 다시말해 len(dense_block_layer_list) = CNN이 사용하는 Dense block의 개수입니다.
    # is_flatten : 연산의 마지막에 1차원 벡터로 만들어주는 nn.Flatten()을 사용할 것인지 말지 결정합니다. 
    model = dense_CNN(input_channel = input_channel, dense_block_first_channel = 64, dense_block_layer_list = [16, 32, 8], is_flatten = True)
    ~~~

3. MLP-Mixer : MLP만 가지고 이미지의 classification을 수행하는 모델입니다. 자세한 설명은 MLP.MLP_Mixer.py에 있습니다. <br> 사용 예)
    ~~~python
    from Linear.MLP_Mixer import *

    # image_size : 입력값으로 넣을 이미지의 크기. 
    # patch_size : MLP-Mixer가 사용할 patch의 크기. MLP-Mixer는 이미지를 patch단위로 나눈 다음 toekn으로 embedding하고 mixer-layer에 넣어줍니다.
    # C : desired hidden dimension. 개인적으로 설정하는 값입니다. 
    # N : MLP-Mixer가 사용할 Mixer-Layer의 개수.
    # classes_num : MLP_Mixer가 분류할 클래스의 개수.

    image_size = (3, 224, 224)
    mixer = MLP_Mixer(input_size=image_size, patch_size=32, C=512, N=8, classes_num=1000) # 입력값으로 들어가는 것들을 수정해서 사용하시면 됩니다
    ~~~