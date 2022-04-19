from tqdm import tqdm
import copy
import torch
from PIL import Image
import numpy as np
import cv2
import os
from terminaltables import AsciiTable

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# top 5-acc 휙득
def get_top_5_accuracy(preds, targets) :
  
    top_5_pred_indices = torch.sort(preds, 1).indices[:,:5] # 이미지 별 예측값 중 가장 점수가 높은 상위 5개씩 추출
    targets_view = targets.view(-1,1) # torch.eq에 사용하기 위해 [-1, 1] 형태로 변환시켜줌
    
    answer_count = torch.eq(top_5_pred_indices, targets_view).type(torch.LongTensor).sum().item() # 전체 배치에서 맞춘 것의 개수 측정

    return (answer_count/targets.size()[0]) # 맞춘 개수를 전체 배치로 나눠 정확도를 계산 후 반환


# 학습 결과 출력 
# train_loss : train dataset으로 얻은 loss
# train_acc :  train dataset으로 얻은 accuracy
def print_result(train_loss, train_acc) :
    print(AsciiTable(
                [
                    ["Type", "Value"],
                    ["train loss", format(round(train_loss, 6), 'f')],
                    ["train acc ", format(round(train_acc, 6), 'f')]
                ]).table.replace('+', '|'))

# train함수. loss가 가장 낮은 모델을 출력
# model : 학습시킬 모델(네트워크)
# EPOCH : 학습에 사용할 epoch
# loss_function : 학습에 사용되는 손실 함수
# dataset_loader : 학습에 사용되는 데이터셋을 지정한 미니배치 단위로 불러올 수 있는 DataLoader 객체
# optimizer : 학습에 사용할 optimizer. Adam, SGD 등
# gradient_clipping : gradient clipping을 할 것인가? 
# device : 연산을 수행할 장치. cpu 혹은 cuda
def train(model, EPOCH, loss_function, optimizer, train_dataset_loader, gradient_clipping = False, device = 'cuda') :
    
    min_loss = None  # 학습 중에 휙득한 loss중 가장 낮은 값
    min_loss_model = None # loss가 가장 낮은 model

    # 코드 구조 출처 : https://github.com/eriklindernoren/PyTorch-YOLOv3

    batches_done = 1
    warmup_threshold = 10000 # linear learning rate warmup을 위한 변수

    # 기존에 가지고 있던 optimizer의 learning rate를 저장
    for g in optimizer.param_groups:
        init_lr = g['lr']

    train_loss_record = []
    train_acc_record = []

    valid_loss_record = []
    valid_acc_record = []

    for epoch in range(EPOCH):        

        model.train() # 모델을 학습 모드로 설정

        pbar_train = tqdm(train_dataset_loader, desc=f"Training Epoch {epoch}")
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        for _, (imgs, targets) in enumerate(pbar_train):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            with torch.autograd.set_detect_anomaly(True) : 
                preds = model(imgs)
                loss = loss_function(preds, targets) # 나중에 loss를 저장 후 print_loss()에서 출력
                train_top_5_acc = get_top_5_accuracy(preds, targets)
                loss.backward()
            if gradient_clipping == True :
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # learning rate warmup을 위한 학습률 조정
            # batches_done이 warmup_threshold이 될 때 까지 학습률을 선형으로 증가시키다가 warmup_threshold부터 optimizer의 기존 학습률을 사용하겠다는 뜻
            if batches_done <= warmup_threshold :
                for g in optimizer.param_groups:
                    g['lr'] = init_lr * (batches_done/warmup_threshold)

            optimizer.step()
            optimizer.zero_grad()

            train_epoch_loss += loss.item()
            train_epoch_acc += train_top_5_acc
            batches_done +=1
        
        # 결과 정리
        train_epoch_loss /= len(train_dataset_loader)
        train_epoch_acc  /= len(train_dataset_loader)
    
        train_loss_record.append(train_epoch_loss)
        train_acc_record.append(train_epoch_acc)

        # 결과 표시
        print_result(train_epoch_loss, train_epoch_acc)

        # 기존의 min_loss를 비교해서 더 낮게 나오면 min_loss를 현재 epoch_loss로 교체하고 현재 epoch에서 학습시킨 모델을 반환할 모델(min_loss_model)로 설정
        if min_loss == None or min_loss < train_epoch_loss :
            min_loss = train_epoch_loss
            min_loss_model = copy.deepcopy(model)

    return min_loss_model


# ImageNet에 있는 데이터를 가지고 이미지 처리 네트워크의 학습을 시키는데 사용하는 dataset 클래스. 
# dataset_root : 데이터셋의 최상위 경로
# image_name_list : 이미지 파일들의 이름이 담겨있는 리스트. 예) ['1.jpg', '2.jpg',...]
# image_size : 네트워크가 받는 이미지의 해상도. 예) (224, 224)
# label list : 이미지별 라벨 데이터들이 담겨있는 리스트. 예) ['dog', 'cat',...]
# classes_list : 데이터셋에 들어있는 클래스를 나열한 리스트. 예) ['dog', 'cat, 'cow',...]
class Train_ImageNet_dataset(Dataset) :
    def __init__(self, dataset_root, image_name_list, image_size, label_list, classes_list): 
        
        self.dataset_root = dataset_root
        self.image_name_list = image_name_list
        self.image_size = image_size
        self.label_list = label_list
        self.classes_list = classes_list
        
    def __len__(self): 
        return len(self.image_name_list) 
    
    def __getitem__(self, index): 
        
        # 이미지 처리=========================
        to_tensor = transforms.ToTensor()
        img = np.array(Image.open(self.dataset_root + self.image_name_list[index]).convert('RGB'))
        img = cv2.resize(img, dsize=(self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_AREA)
        # Normalize 연산할 때 기존 방식인 '255로 나누기'를 수행하는 과정에서 NaN이 발생. opencv서서 제공하는 함수로 normalize 처리하니 NaN이 발생하지 않음 
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = to_tensor(img) # tensor로 만들어서 네트워크에서 처리할 수 있게 만든다. 
        # 이미지 처리=========================
        
        # label data
        label = self.classes_list.index(self.label_list[index])
        
        return img, torch.tensor(label) # input, target 모두 tensor로 반환

# Train_ImageNet_dataset의 DataLoader를 만드는 메소드
# dataset_root : 데이터셋의 최상위 경로
# image_name_list : 이미지 파일들의 이름이 담겨있는 리스트. 예) ['1.jpg', '2.jpg',...]
# image_size : 네트워크가 받는 이미지의 해상도. 예) (224, 224)
# label list : 이미지별 라벨 데이터들이 담겨있는 리스트. 예) ['dog', 'cat',...]
# classes_list : 데이터셋에 들어있는 클래스를 나열한 리스트. 예) ['dog', 'cat, 'cow',...]
# num_workers : DataLoader에서 미니배치 단위로 데이터를 불러올 때 사용할 프로세서의 개수. 사용할 쓰레드의 개수로 생각하면 되겠다. 
# BATCH_SIZE : DataLoader가 사용할 mini batch의 크기
def make_Train_ImageNet_dataloader(dataset_root, image_name_list, image_size, label_list, classes_list, num_workers, BATCH_SIZE = 16) :
    dataset = Train_ImageNet_dataset(dataset_root, image_name_list, image_size, label_list, classes_list)
    dataloader = DataLoader(
        dataset = dataset, # 사용할 데이터셋
        batch_size = BATCH_SIZE, # 미니배치 크기
        shuffle=True, # 에포크마다 데이터셋 셔플할건가? 
        num_workers=num_workers,
        pin_memory=False,
        # drop_last : 우리가 batch 단위로 데이터를 불러올 때 마지막 batch는 BATCH_SIZE보다 작을 수 있다. 이를 사용하지 않을려면 True를, 사용하려면 False를 설정하면 된다.
        drop_last=True) 
    return dataloader

# ImageNet-2012 dataset의 Train Dataloader를 얻는 함수
# dataset_root : 데이터셋의 최상위 경로
# image_size : 네트워크가 받는 이미지의 해상도. 예) (224, 224)
# num_workers : DataLoader에서 미니배치 단위로 데이터를 불러올 때 사용할 프로세서의 개수. 사용할 쓰레드의 개수로 생각하면 되겠다. 
# BATCH_SIZE : DataLoader가 사용할 mini batch의 크기
def get_ImageNet_Train_DataLoader(dataset_root, image_size, num_workers, BATCH_SIZE) :
    image_name_list = os.listdir(dataset_root)
    label_list = []
    for image_name in image_name_list :
        class_name = image_name.split('_')[0] # ImageNet 기준이며 
        label_list.append(class_name)
    classes_list = list(set(label_list))
    
    train_dataloader = make_Train_ImageNet_dataloader(dataset_root = dataset_root, image_name_list = image_name_list, 
                                       image_size = image_size, label_list = label_list, classes_list = classes_list, 
                                       num_workers = num_workers, BATCH_SIZE = BATCH_SIZE)
    
    return train_dataloader, len(classes_list)