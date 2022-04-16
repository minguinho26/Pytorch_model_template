# Pytorch_model_template

제작자 : 김민규 <br>
리포지토리 생성일 : 22.4.16

## 설명
파이토치를 이용해 제작한 네트워크(MLP, CNN 등)들을 쉽게 사용하기 위해 제작하는 템플릿. <br>
22.4.16 기준으로 3가지 유형(Linear, CNN, residual CNN)의 네트워크를 구현했습니다. 

1. Linear : 선형 연산을 수행하는 nn.Linear 레이어를 가지고 구현했습니다. <br> 사용 예)
    ~~~python
    from Linear.normal_MLP import *

    input_size = 64
    neural_list = [64, 64, 128, 128]

    # 기본적으로 연산 도중에 사용하는 활성화 함수는 ReLU, 연산의 마지막에 사용하는 활성화 함수는 Sigmoid 함수로 설정했습니다. 그리고 Batch Normalization도 사용하게 설정했습니다. 이 설정들은 초기화 할 때 입력하는 값을 통해 개인적으로 변경하실 수 있습니다. 

    model = normal_MLP(input_size, neural_list)
    # 설정을 변경한 예
    model = normal_MLP(input_size, neural_list, mid_activation_func = 'leaky_relu',  last_activateion_func = 'softmax', batch_normalization_use = False)
    ~~~

2. CNN : Covolution 연산을 수행하는 네트워크입니다. <br> 사용 예)
    ~~~python
    # 일반적인 CNN을 사용하는 경우
    from CNN.normal_CNN import *

    input_size : (3, 224, 224)
    kernal_list : [64, 64, 'p', 128, 128] # p는 pooling을 나타냅니다. 

    # 기본적으로 연산 도중에 사용할 활성화 함수로 ReLU를 사용하고 연산의 마지막에 flatten 연산을 수행하지 않게 설정했습니다. 이 역시 초기화시 입력하는 값을 통해 수정할 수 있습니다. 그리고 Batch Normalization도 기본적으로 사용하게끔 설정되어 있으나 역시 수정 가능합니다. 

    model = normal_CNN(input_size, kernal_list)
    # 설정을 변경한 예
    model = normal_CNN(input_size, kernal_list, activation_func = 'leaky_relu',  is_flatten = True, batch_normalization_use = False)
    ~~~

    ~~~python
    # Residual CNN, 그러니까 ResNet 계열의 CNN을 사용하는 경우
    from CNN.residual_CNN import *

    input_size : (3, 224, 224)
    kernal_list : [64, 64, 128, 128] # residual_CNN은 커널이 변경될 때마다 pooling을 수행합니다.  

    # ResNet 논문에 따르면 50층 이상의 큰 네트워크를 설계할 때는 3개의 Convolutional layer로 구성된 residual block을 쓰고 그보다 작은 네트워크를 설계할 때는 2개의 Convolutional layer로 구성된 residual block을 사용합니다. residual_CNN은 기본적으로 2게의 Convolutional layer로 구성된 residual block을 사용하게끔 설정되어 있으나 초기화시 입력하는 값을 통해 변경 가능합니다. 그리고 flatten에 관한 설정도 수행하지 않는게 기본 설정이지만 역시 변경 가능합니다. 

    model = residual_CNN(input_size, kernal_list)
    # 설정을 변경한 예
    model = residual_CNN(input_size, kernal_list, Residual_Block_size = 'big', is_flatten = True)
    ~~~