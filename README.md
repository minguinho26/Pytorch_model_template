# Pytorch_model_template

Note : 현재 코드 수정 후 성능 확인을 하고 있습니다. 추후 수정된 코드와 함께 결과를 업데이트 하겠습니다.

제작자 : 김민규 <br>
리포지토리 생성일 : 22.4.16

## 설명
파이토치를 이용해 제작한 네트워크(MLP, CNN 등)들을 쉽게 사용하기 위해 제작하는 템플릿. <br>
22.4.16 기준으로 3가지 유형(Linear, CNN, residual CNN)의 네트워크를 구현했습니다. 

* util.py : 모델 학습에 사용되는 함수들을 모았습니다. 자세한 내용은 파일에 적혀있는 주석을 참고하시면 됩니다.

## 구성(22.4.18 기준)
📁 main <br>
└📁Basic_model <br>
⠀└normal_MLP.py <br>
└📁Image_Classification <br>
⠀└📁CNN <br>
⠀⠀└normal_CNN.py <br>
⠀⠀└dense_CNN.py <br>
⠀⠀└residual_CNN.py <br>
⠀└📁Linear <br>
⠀⠀└MLP_Mixer.py <br>
└📁util <br>
⠀└util.py <br>

### **<1> Basic model**
가장 기본적인 MLP가 있습니다. 

1. Linear : 선형 연산을 수행하는 nn.Linear 레이어를 가지고 구현했습니다. <br> 사용 예)
    ~~~python
    from Basic_model.normal_MLP import *

    # 기본적으로 연산 도중에 사용하는 활성화 함수는 ReLU, 연산의 마지막에 사용하는 활성화 함수는 Sigmoid 함수로 설정했습니다. 그리고 Batch Normalization도 사용하게 설정했습니다. 이 설정들은 초기화 할 때 입력하는 값을 통해 개인적으로 변경하실 수 있습니다. 
    model = normal_MLP(input_size = 64, neural_list = [64, 64, 128, 128],  mid_activation_func = 'leaky_relu',  last_activateion_func = 'softmax', batch_normalization_use = False)
    ~~~

### **<2> Image Classification**
이미지에 존재하는 객체의 클래스를 분류하는 일을 수행하는 네트워크를 구현했습니다. 

#### **<2-1> CNN**
CNN으로 구현한 네트워크들입니다. 

1. normal_CNN : Covolution 연산을 수행하는 CNN 중 가장 기본적인 형태를 가진 네트워크입니다. <br> 
   사용 예)
    ~~~python
    # 일반적인 CNN을 사용하는 경우
    from Image_Classification.CNN.normal_CNN import *

    # input_size : 입력되는 feature의 크기입니다. 예) (3, 224, 224)
    # kernal_list : Convolutional layer들이 사용할 커널의 개수. p는 pooling을 나타냅니다. 
    # num_classes : 분류할 클래스 개수를 나타냅니다
    # batch_normalization_use : 연산 중간중간에 Batch Normalizatoin을 사용할지 말지 결정합니다.
    # device : 네트워크를 가지고 연산할 장치를 선택합니다.

    # 사용 예
    model = normal_CNN(input_size = (3, 224, 224), kernal_list = [64, 64, 'p', 128, 128], num_classes = 1000, activation_func = 'leaky_relu', batch_normalization_use = False, device = 'cuda')
    ~~~
2. residual_CNN : Skip connection을 수행하는 CNN입니다. ResNet에서 제안한 residual block을 사용했습니다.
    ~~~python
    # Residual CNN, 그러니까 ResNet 계열의 CNN을 사용하는 경우
    from Image_Classification.CNN.residual_CNN import *

    # input_size : 입력되는 feature의 크기입니다. 예) (3, 224, 224) 
    # kernal_list : Convolutional layer들이 사용할 커널의 개수. residual_CNN은 커널이 변경될 때마다 pooling을 수행합니다.  
    # num_classes : 분류할 클래스 개수를 나타냅니다
    # Residual_Block_size : small이면 2개의 Convolutional layer로 구성된 residual block 사용, big이면 3개의 Convolutional layer로 구성된 residual block 사용
    # device : 네트워크를 가지고 연산할 장치를 선택합니다.

    # ResNet 논문에 따르면 50층 이상의 큰 네트워크를 설계할 때는 3개의 Convolutional layer로 구성된 residual block을 쓰고 그보다 작은 네트워크를 설계할 때는 2개의 Convolutional layer로 구성된 residual block을 사용합니다. 
    model = residual_CNN(input_size = (3, 224, 224), kernal_list = [64, 64, 128, 128], num_classes = 1000, Residual_Block_size = 'big', device = 'cuda')
    ~~~

3. dense_CNN : DenseNet에서 제안한 Dense block을 사용해 구성한 CNN입니다. 
    ~~~python
    # Dense CNN, DenseNet을 구성하는 Dense block으로 구성된 CNN
    from CNN.dense_CNN import *

    # input_size : 입력되는 feature의 크기입니다. 예) (3, 224, 224)
    # dense_block_first_channel. 맨처음 Dense Block이 사용하는 커널의 개수. DenseNet은 다음 Dense block으로 넘어갈 때마다 채널의 크기를 2배씩 늘립니다. 그래서 맨처음 channel, 즉 kernal의 값만 받습니다.
    # dense_block_layer_list. 각 Dense block에서 쓰이는 residual block의 개수를 나타냅니다. 다시말해 len(dense_block_layer_list) = CNN이 사용하는 Dense block의 개수입니다.
    # is_flatten : 연산의 마지막에 1차원 벡터로 만들어주는 nn.Flatten()을 사용할 것인지 말지 결정합니다. 
    model = dense_CNN(input_size = (3, 224, 224), dense_block_first_channel = 64, dense_block_layer_list = [16, 32, 8], num_classes = 1000, device = 'cuda')
    ~~~

#### **<2-2> Linear**
MLP로 구현한 네트워크입니다.

1. MLP-Mixer : MLP만 가지고 이미지의 classification을 수행하는 모델입니다. 자세한 설명은 MLP.MLP_Mixer.py에 있습니다. <br> 사용 예)
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