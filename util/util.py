from tqdm import tqdm
import torch
import os
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision


# top 5-acc 휙득
def get_top_k_accuracy(preds, targets, k) :
  
    top_k_pred_indices = torch.topk(preds, k).indices
    targets_view = targets.view(-1,1) # torch.eq에 사용하기 위해 [-1, 1] 형태로 변환시켜줌
    
    answer_count = torch.eq(top_k_pred_indices, targets_view).type(torch.LongTensor).sum().item() # 전체 배치에서 맞춘 것의 개수 측정
    
    return (answer_count/targets.size()[0]) # 맞춘 개수를 전체 배치로 나눠 정확도를 계산 후 반환


# 학습 결과 출력 
# train_loss : train dataset으로 얻은 loss
# train_acc :  train dataset으로 얻은 accuracy
def print_result(train_loss, train_acc, test_loss, test_acc) :
    print(AsciiTable(
                [
                    ["Type", "Value"],
                    ["train loss", format(round(train_loss, 6), 'f')],
                    ["train acc ", format(round(train_acc, 6), 'f')],
                    ["test loss", format(round(test_loss, 6), 'f')],
                    ["test acc ", format(round(test_acc, 6), 'f')]
                ]).table.replace('+', '|'))

# train함수. loss가 가장 낮은 모델을 출력
# model : 학습시킬 모델(네트워크)
# EPOCH : 학습에 사용할 epoch
# loss_function : 학습에 사용되는 손실 함수
# dataset_loader : 학습에 사용되는 데이터셋을 지정한 미니배치 단위로 불러올 수 있는 DataLoader 객체
# optimizer : 학습에 사용할 optimizer. Adam, SGD 등
# gradient_clipping : gradient clipping을 할 것인가? 
# device : 연산을 수행할 장치. cpu 혹은 cuda
def train(model, EPOCH, loss_function, optimizer_name, train_dataset_loader, test_dataset_loader, k, gradient_clipping = False, linear_burn_up = False, device = 'cuda', see_result_during_training = False) :
    
    if optimizer_name == 'adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    min_loss = None  # 학습 중에 휙득한 loss중 가장 낮은 값

    # 코드 구조 출처 : https://github.com/eriklindernoren/PyTorch-YOLOv3

    batches_done = 1
    warmup_threshold = len(train_dataset_loader) * 2 # linear learning rate warmup을 위한 변수

    # 기존에 가지고 있던 optimizer의 learning rate를 저장
    for g in optimizer.param_groups:
        init_lr = g['lr']

    train_loss_record = []
    train_acc_record = []
    
    test_loss_record = []
    test_acc_record = []

    model.train() # 모델을 학습 모드로 설정
    
    print('====training start====')
    for epoch in range(EPOCH):        

        pbar_train = tqdm(train_dataset_loader, desc=f"Training Epoch {epoch}")
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        for batch_idx, (imgs, targets) in enumerate(pbar_train):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            preds = model(imgs)
            
            loss = loss_function(preds, targets) # 나중에 loss를 저장 후 print_loss()에서 출력
            train_top_k_acc = get_top_k_accuracy(preds.clone().detach().requires_grad_(False), targets, k)

            loss.backward()
            # learning rate warmup을 위한 학습률 조정
            # batches_done이 warmup_threshold이 될 때 까지 학습률을 선형으로 증가시키다가 warmup_threshold부터 optimizer의 기존 학습률을 사용하겠다는 뜻

            for g in optimizer.param_groups:
                if (batches_done <= warmup_threshold) and linear_burn_up == True :
                    g['lr'] = init_lr * (batches_done/warmup_threshold)
                else :
                    g['lr'] = init_lr

            
            if gradient_clipping == True :
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            
            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

            train_epoch_loss += loss.item()
            train_epoch_acc += train_top_k_acc
            batches_done +=1
            
        pbar_test = tqdm(test_dataset_loader, desc=f"Evaluation Epoch {epoch}")   
        test_epoch_loss = 0.0
        test_epoch_acc = 0.0
        
        model.eval()
        for batch_idx, (imgs, targets) in enumerate(pbar_test):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            with torch.no_grad():
            
                preds = model(imgs)
            
                loss = loss_function(preds, targets) # 나중에 loss를 저장 후 print_loss()에서 출력
                test_top_k_acc = get_top_k_accuracy(preds.clone().detach().requires_grad_(False), targets, k)

            test_epoch_loss += loss.item()
            test_epoch_acc += test_top_k_acc
            
        model.train()
        
        # 결과 정리
        train_epoch_loss /= len(train_dataset_loader)
        train_epoch_acc  /= len(train_dataset_loader)
        
        test_epoch_loss /= len(test_dataset_loader)
        test_epoch_acc  /= len(test_dataset_loader)
        
        train_loss_record.append(train_epoch_loss)
        train_acc_record.append(train_epoch_acc)
        
        test_loss_record.append(test_epoch_loss)
        test_acc_record.append(test_epoch_acc)
        
        # 결과 표시를 하겠다고 하면 결과 보여줌
        if see_result_during_training == True :
            print_result(train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc)

        # 기존의 min_loss를 비교해서 더 낮게 나오면 min_loss를 현재 epoch_loss로 교체하고 현재 epoch에서 학습시킨 모델을 반환할 모델(min_loss_model)로 설정
        if min_loss == None or min_loss < test_epoch_loss :
            min_loss = train_epoch_loss
            torch.save(model.state_dict(), './best_model.pt')
    
    model.load_state_dict(torch.load('./best_model.pt'))
    os.remove('./best_model.pt')
    
    del imgs, targets, preds, loss, test_top_k_acc
    torch.cuda.empty_cache()
    
    print('====training end====')
    
    return model, [train_loss_record, train_acc_record, test_loss_record, test_acc_record]


def plot_result(data_list) :
    
    title_list = ['Train Loss', 'Train Acc', 'Test Loss', 'Test Acc']
    
    for i in range(4) :
        
        plt.figure(figsize=(20,10))
        plt.suptitle(title_list[i], fontsize=32)
        plt.plot(data_list[0][i], 'c--', label = 'normal_cnn')
        plt.plot(data_list[1][i], 'r-', label = 'residual_cnn')
        plt.plot(data_list[2][i], 'g-.', label = 'mlp_mixer_s4')
        plt.plot(data_list[3][i], 'k-', label = 'mlp_mixer_s2')
        plt.plot(data_list[4][i], 'm--.', label = 'mlp_mixer_b4')
        plt.plot(data_list[5][i], 'b-.', label = 'mlp_mixer_b2')
        plt.legend(prop={'size': 15}) # 범례 표시
        plt.show()
        print('')

# cifar dataset 휙득
# num_classes : 데이터셋에 있는 클래스의 개수. 10 혹은 100을 입력
# batch_size : DataLoader가 사용할 미니배치의 크기
def get_cifar_dataset(classes_num = 10, batch_size = 512) :
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if classes_num == 10 :
        trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./cifar', train=False,
                                               download=True, transform=transform)
    elif classes_num == 100 :
        trainset = torchvision.datasets.CIFAR100(root='./cifar', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./cifar', train=False,
                                               download=True, transform=transform)
    
    
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=6,pin_memory=False,
                                                  drop_last=True)
    
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=6,pin_memory=False,
                                                  drop_last=True)
    
    return train_dataloader, test_dataloader, trainset[0][0].size()
