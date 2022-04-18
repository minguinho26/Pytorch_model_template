from tqdm import tqdm
import copy
import torch
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# dataset, train 등 사용에 필요한 메소드들을 구현

# train함수. loss가 가장 낮은 모델을 출력
# model : 학습시킬 모델(네트워크)
# EPOCH : 학습에 사용할 epoch
# loss_function : 학습에 사용되는 손실 함수
# dataset_loader : 학습에 사용되는 데이터셋을 지정한 미니배치 단위로 불러올 수 있는 DataLoader 객체
# optimizer : 학습에 사용할 optimizer. Adam, SGD 등
# device : 연산을 수행할 장치. cpu 혹은 cuda
def train(model, EPOCH, loss_function, optimizer, dataset_loader, device = 'cuda') :

    model.train() # 모델을 학습 모드로 설정
    
    min_loss = None  # 학습 중에 휙득한 loss중 가장 낮은 값
    min_loss_model = None # loss가 가장 낮은 model

    # 코드 구조 출처 : https://github.com/eriklindernoren/PyTorch-YOLOv3
    for epoch in range(EPOCH):        
        pbar = tqdm(dataset_loader, desc=f"Training Epoch {epoch}")
        
        epoch_loss = 0.0
        for _, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            with torch.autograd.set_detect_anomaly(True) : 
                preds = model(imgs)
                loss = loss_function(preds, targets) # 나중에 loss를 저장 후 print_loss()에서 출력

                loss.backward()
                
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss /= len(dataset_loader)
        
        # 다음 에포크로 넘어가기 전에 현재 에포크에서 얻은 epoch_loss를 출력
        pbar.set_description(f"Training Epoch {epoch}, epoch_loss : {epoch_loss}")

        # 기존의 min_loss를 비교해서 더 낮게 나오면 min_loss를 현재 epoch_loss로 교체하고 현재 epoch에서 학습시킨 모델을 반환할 모델(min_loss_model)로 설정
        if min_loss == None or min_loss < epoch_loss :
            min_loss = epoch_loss
            min_loss_model = copy.deepcopy(model)
    return min_loss_model


# 이미지 처리를 위한 네트워크 학습을 위한 dataset 클래스 
# dataset_root : 데이터셋의 최상위 경로
# image_name_list : 이미지 파일들의 이름이 담겨있는 리스트. 예) ['1.jpg', '2.jpg',...]
# image_size : 네트워크가 받는 이미지의 해상도. 예) (224, 224)
# label list : 이미지별 라벨 데이터들이 담겨있는 리스트. 예) ['dog', 'cat',...]
# classes_list : 데이터셋에 들어있는 클래스를 나열한 리스트. 예) ['dog', 'cat, 'cow',...]
class image_dataset(Dataset) :
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

# image_dataset의 DataLoader를 만드는 메소드
# dataset_root : 데이터셋의 최상위 경로
# image_name_list : 이미지 파일들의 이름이 담겨있는 리스트. 예) ['1.jpg', '2.jpg',...]
# image_size : 네트워크가 받는 이미지의 해상도. 예) (224, 224)
# label list : 이미지별 라벨 데이터들이 담겨있는 리스트. 예) ['dog', 'cat',...]
# classes_list : 데이터셋에 들어있는 클래스를 나열한 리스트. 예) ['dog', 'cat, 'cow',...]
# num_workers : DataLoader에서 미니배치 단위로 데이터를 불러올 때 사용할 프로세서의 개수. 사용할 쓰레드의 개수로 생각하면 되겠다. 
# BATCH_SIZE : DataLoader가 사용할 mini batch의 크기
def make_image_dataloader(dataset_root, image_name_list, image_size, label_list, classes_list, num_workers, BATCH_SIZE = 16) :
    dataset = image_dataset(dataset_root, image_name_list, image_size, label_list, classes_list)
    dataloader = DataLoader(
        dataset = dataset, # 사용할 데이터셋
        batch_size = BATCH_SIZE, # 미니배치 크기
        shuffle=True, # 에포크마다 데이터셋 셔플할건가? 
        num_workers=num_workers,
        pin_memory=False,
        # drop_last : 우리가 batch 단위로 데이터를 불러올 때 마지막 batch는 BATCH_SIZE보다 작을 수 있다. 이를 사용하지 않을려면 True를, 사용하려면 False를 설정하면 된다.
        drop_last=True) 
    return dataloader
