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

    # 사용 예
    image_size = (3, 224, 224) # 네트워크에 넣을 이미지의 크기
    kernal_list_normal_CNN = [16, 16,'p', 32, 32, 'p', 64, 64, 'p', 128, 128, 'p', 256, 256] # 숫자는 Conv2d의 채널 개수, 'p'는 pooling을 의미합니다
    classes_num = 10 # 분류할 클래스 개수

    normal_cnn = normal_CNN(input_size = image_size, kernal_list = kernal_list_normal_CNN, num_classes = classes_num, activation_func = 'relu', batch_normalization_use = True, device = 'cuda')


    ~~~
2. residual_CNN : Skip connection을 수행하는 CNN입니다. ResNet에서 제안한 residual block을 사용했습니다.
    ~~~python
    # Residual CNN, 그러니까 ResNet 계열의 CNN을 사용하는 경우
    from Image_Classification.CNN.residual_CNN import *

    # 사용 예
    image_size = (3, 224, 224)
    kernal_list_residual_CNN = [32, 32, 64, 64, 64, 64, 128, 128, 128]
    classes_num = 10
    
    residual_cnn = residual_CNN(input_size = image_size, kernal_list = kernal_list_residual_CNN, num_classes = classes_num, Residual_Block_size = 'small')

    ~~~


#### **<2-2> Linear**
MLP로 구현한 네트워크입니다.

1. MLP-Mixer : MLP만 가지고 이미지의 classification을 수행하는 모델입니다. 자세한 설명은 MLP.MLP_Mixer.py에 있습니다. <br> 사용 예)
    ~~~python
    from Linear.MLP_Mixer import *

    # 사용 예
    image_size = (3, 224, 224)
    mlp_mixer = MLP_Mixer(input_size=image_size, patch_size=4, C=128, N=6, classes_num=classes_num).to('cuda')
    ~~~

## 실험 세팅

아래 조건을 기반으로 성능을 평가하는 실험을 수행했습니다. 
> 실험을 수행한 코드는 model.ipynb이므로 자세한 정보는 해당 주피터 노트북  파일에서 확인하실 수 있습니다.

<br>

### Training setting

~~~python
# Optimizer : Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# Gradient Cliping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# Epoch
EPOCH = 600
# Loss function
loss_function = torch.nn.CrossEntropyLoss()
~~~

<br>

### Dataset : CIFAR 10(train, test)
~~~python
batch_size = 500

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6,pin_memory=False, drop_last=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6,pin_memory=False, drop_last=True)
~~~

<br>

### Model 정보

1. normal_cnn : 일반적인 구조를 가지는 CNN
2. residual_cnn : Residual block으로 설계된 CNN
3. mlp_mixer_S4 : Image patch가 (4,4)고 C를 128로 설정한 MLP-Mixer
4. mlp_mixer_S2 : Image patch가 (2,2)고 C를 128로 설정한 MLP-Mixer
5. mlp_mixer_B4 : Image patch가 (4,4)고 C를 192로 설정한 MLP-Mixer
6. mlp_mixer_B2 : Image patch가 (2,2)고 C를 192로 설정한 MLP-Mixer

<br>

### Model Parameter 정보

| model        | parameter_num |
|--------------|---------------|
| normal_cnn   | 1,183,322       |
| residual_cnn | 1,134,026       |
| mlp_mixer_S4 | 273,674        |
| mlp_mixer_S2 | 1,062,026       |
| mlp_mixer_B4 | 698,186        |
| mlp_mixer_B2 | 1,756,106       |

<br>

## 실험 결과

<img width="910" alt="스크린샷 2022-04-30 오후 5 12 02" src="https://user-images.githubusercontent.com/50979281/166097671-4b8d3b73-d323-4557-9fff-c0a1decaad4f.png">

<br>

<img width="913" alt="스크린샷 2022-04-30 오후 5 12 13" src="https://user-images.githubusercontent.com/50979281/166097668-07357d0d-66db-4359-9a53-357a870d4ae1.png">

normal_cnn이 제일 좋은 성능을 보여줍니다. CIFAR10 데이터셋의 크기가 작아서 그런 것으로 추정하고 있습니다.
