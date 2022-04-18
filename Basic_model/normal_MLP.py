import torch.nn as nn

# 일반적인 MLP. 
# last_activateion_func : 마지막에 사용할 활성화 함수. sigmoid, softmax 등. 이외의 문자열을 넣을 경우 활성화함수 추가하지 않음
# mid_activation_func : 네트워크의 중간중간에 사용할 활성화 함수. relu, leaky_relu 등. 이외의 문자열을 넣을 경우 활성화함수 추가하지 않음
class normal_MLP(nn.Module) :

    def __init__(self, input_size, neural_list, mid_activation_func = 'relu',  last_activateion_func = 'sigmoid', batch_normalization_use = True) :
        super(normal_MLP, self).__init__()

        self.Linear = nn.Sequential()

        for i in range(len(neural_list)) :

            # Linear Layer 추가
            if i == 0 :
                self.Linear.add_module("Linear_" + str(i), nn.Linear(input_size, neural_list[i], bias = False))
            else :
                self.Linear.add_module("Linear_" + str(i), nn.Linear(neural_list[i - 1], neural_list[i], bias = False))
            
            # BN 사용 여부
            if batch_normalization_use == True :
                self.Linear.add_module("BN_" + str(i), nn.BatchNorm1d(neural_list[i], affine = False))
            
            if i != len(neural_list) - 1 :
                if mid_activation_func.lower() == 'relu' : 
                    self.Linear.add_module('ReLU_' + str(i), nn.ReLU())
                elif mid_activation_func.lower() == 'leaky_relu' : 
                    self.Linear.add_module('LeakyReLU_' + str(i), nn.LeakyReLU())
        
        if last_activateion_func.lower() == 'sigmoid' : 
                self.Linear.add_module('sigmoid', nn.Sigmoid())
        elif last_activateion_func.lower() == 'softmax' :
            self.Linear.add_module('softmax', nn.Softmax(dim = 1))
    
    def forward(self, x) :
        return self.Linear(x)