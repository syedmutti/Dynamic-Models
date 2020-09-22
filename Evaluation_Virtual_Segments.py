

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import numpy as np
import os


class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
       

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelH2LL, modelH2UL, modelH2LA, modelH2UA):
        super(MyEnsemble, self).__init__()
        
        self.model_lower_leg = modelA
        self.model_upper_leg = modelB
        self.model_lower_arm = modelC
        self.model_upper_arm = modelD
        
        # Models for Virtual Reconstruction
        
        self.height_to_lowerleg = modelH2LL
        self.height_to_upper_leg = modelH2UL
        self.height_to_lower_arm = modelH2LA
        self.height_to_upper_arm = modelH2UA
        

                
    def forward(self, x1, flag):
        
        def reconstruct_Segments(height):

            pred_LL  = self.height_to_lowerleg(height/201.513) * 49.319  # 201.513741064 493.195872265
            pred_UL = self.height_to_upper_leg(height/201.913) * 58.554   # 201.913788186 585.546772618
            pred_LA = self.height_to_lower_arm(height/204.805) * 31.670  # 204.805580317 316.702168533
            pred_UA = self.height_to_upper_arm(height/204.104) * 42.5100   # 204.104986515 425.100857276

            return pred_LL, pred_UL, pred_LA, pred_UA
        
        if flag == 'lower_leg':
            x = self.model_lower_leg(x1)
            x*=195.5    
            
        if flag == 'upper_leg':
            x = self.model_upper_leg(x1)
            x*=204.1
            
        if flag == 'lower_arm':
            x = self.model_lower_arm(x1)
            x*=204.68
            
        if flag == 'upper_arm':
            x = self.model_upper_arm(x1)
            x*= 209.2665
            
        #print('Height predicted from {} : is {} cm'.format(flag, x.item()))
        
        pred_LL, pred_UL, pred_LA, pred_UA = reconstruct_Segments(x)
        
        return x, pred_LL, pred_UL, pred_LA, pred_UA



# Create models and load state_dicts    

model_lower_leg = MyModelA()
model_upper_leg = MyModelA()
model_lower_arm = MyModelA()
model_upper_arm = MyModelA()

#Virtual Segments Reconstruction Model

model_height_to_lowerleg = MyModelA()
model_height_to_upper_leg = MyModelA()
model_height_to_lower_arm = MyModelA()
model_height_to_upper_arm = MyModelA()

# Load state dicts

# Load state dicts
PATH_lower_leg = 'Trained_models/lower_leg_pytorch_/ansur_model_2.6701'
PATH_upper_leg = 'Trained_models/upper_leg_model_pytorch_/upper_leg_5.84564208984375'
PATH_lower_arm = 'Trained_models/lower_arm_pytorch_/lower_arm_pytorch_3.489548921585083'
PATH_upper_arm = 'Trained_models/upper_arm_pytorch_/upper_arm_pytorch4.744106292724609'

# Virtual Models

PATH_model_height_to_lowerleg =  'Virtual_Segments/Trained_models/height_lower_leg_5.075006484985352'
PATH_model_height_to_upper_leg = 'Virtual_Segments/Trained_models/height_upper_leg_18.44318962097168'
PATH_model_height_to_lower_arm = 'Virtual_Segments/Trained_models/height_lower_arm_9.933013916015625'
PATH_model_height_to_upper_arm = 'Virtual_Segments/Trained_models/Height_upper_arm_9.007071495056152'

# Loading Learned Weights

model_lower_leg.load_state_dict(torch.load(PATH_lower_leg))
model_upper_leg.load_state_dict(torch.load(PATH_upper_leg))
model_lower_arm.load_state_dict(torch.load(PATH_lower_arm))
model_upper_arm.load_state_dict(torch.load(PATH_upper_arm))

# Loading Learned Weights for Virtual Segments

model_height_to_lowerleg.load_state_dict(torch.load(PATH_model_height_to_lowerleg))
model_height_to_upper_leg.load_state_dict(torch.load(PATH_model_height_to_upper_leg))
model_height_to_lower_arm.load_state_dict(torch.load(PATH_model_height_to_lower_arm))
model_height_to_upper_arm.load_state_dict(torch.load(PATH_model_height_to_upper_arm))


#Load Model
model = MyEnsemble(model_lower_leg, model_upper_leg, model_lower_arm, model_upper_arm,
                    model_height_to_lowerleg, model_height_to_upper_leg,
                    model_height_to_lower_arm, model_height_to_upper_arm)

# Display Output 

def display(output):
    x, pred_LL, pred_UL, pred_LA, pred_UA = output
    
    print('Stature: ', x.item(), 'LL: ', pred_LL.item(),'UL: ', pred_UL.item(),'LA: ', pred_LA.item(),'UA: ', pred_UA.item())
    


# # Evaluation Lower Leg (Tibia)


import pandas as pd

dataset_path ='data/ansur_result.csv'
column_names = column_names = ['gender','butt_height','buttock-knee_length','buttock-popliteal_length',
                               'crotch_height','knee_height_midpatella','knee_height_sitting', 
                               'lateral-femoral_epicondyle_height', 'popliteal_height',
                               'stature', 'trochanterion_height', 'wrist_height' ]


raw_dataset = pd.read_csv(dataset_path, names=column_names)
dataset = raw_dataset.copy()

x = dataset['knee_height_midpatella']

x_train = []
y_pred = []

for index, value in enumerate(x):
    if (value != 'knee_height_midpatella') and float(value) > 375 and float(value) < 600 :
        x_train.append(float(value))
    
x_train_float = np.array(x_train[1:len(x_train)-1500])
print((len(x_train_float), len(x_test_float)))

mae = 0
mse = 0
for index, tibia in enumerate(x_train_float):
    
    lower_leg = tibia
    flag = 'lower_leg' 

    x1 = torch.from_numpy(np.array([lower_leg], dtype='float32'))
    x1.view((1,1))
    x, pred_LL, pred_UL, pred_LA, pred_UA = model(x1/592.0, flag)
    mae += abs(pred_LL - tibia/10)
    mse += (pred_LL - tibia/10)**2
    
mae/=index+1
mse/=index+1

print('Mean Absolute Error on ', index, 'samples :', mae.item() )
print('Mean Squared Error on ', index, 'samples :', np.sqrt(mse.item()) )


# # Evaluation Upper Leg (Femur) 


dataset_path ='data/nhannes.csv'
column_names = ['INDEX', 'HEIGHT', 'UPPER_LEG']


raw_dataset = pd.read_csv(dataset_path, names=column_names)

dataset = raw_dataset.copy()

x = dataset['UPPER_LEG']
y = dataset['HEIGHT']

x_train = []

for index, value in enumerate(x):
    if (value != 'upper_leg') and (float(value) > 10.0):
        x_train.append(float(value))

    
x_train_float = np.array(x_train[1:len(x_train)-3000])
print('Samples: ', len(x_train_float))


mae = 0
mse = 0
for index, femur in enumerate(x_train_float):
    
    upper_leg = femur
    flag = 'upper_leg'  # 174.8

    x1 = torch.from_numpy(np.array([upper_leg], dtype='float32'))
    x1.view((1,1))
    x, pred_LL, pred_UL, pred_LA, pred_UA = model(x1/55.5, flag)  
    mae += abs(pred_UL - femur)
    mse += (pred_UL - femur)**2
    
mae/=index+1
mse/=index+1

print('Mean Absolute Error on ', index, 'samples :', mae.item())
print('Mean Squared Error on ', index, 'samples :', np.sqrt(mse.item()))


# # Evaluation Lower Arm (Ulna)


mu , sigma = 250, 7

x_train_ulna = np.random.normal(mu, sigma, 15000)

noise = np.random.normal(0, 5, x_train_ulna.shape)
x_train_ulna = x_train_ulna + noise

print(len(x_train_ulna))


mae = 0
mse = 0
for index, ulna in enumerate(x_train_ulna):
    
    lower_arm = ulna
    flag = 'lower_arm'  

    x1 = torch.from_numpy(np.array([lower_arm], dtype='float32'))

    x1.view((1,1))    
    x, pred_LL, pred_UL, pred_LA, pred_UA = model(x1/280.15, flag)
    mae += abs(pred_LA - ulna/10)
    mse += (pred_LA - ulna/10)**2
    
mae/=index+1
mse/=index+1

print('Mean Absolute Error on ', index, 'samples :', mae.item())
print('Mean Squared Error on ', index, 'samples :', np.sqrt(mse.item()))


# # Evaluation Upper Arm (Humerus)


mu , sigma = 294, 5

x_train_humerus = np.random.normal(mu, sigma, 15000)

noise = np.random.normal(0, 5, x_train_humerus.shape)
x_train_humerus = x_train_humerus + noise

print(len(x_train_humerus))


mae = 0
mse = 0
for index, humerus in enumerate(x_train_humerus):
    upper_arm = humerus
    flag = 'upper_arm'  # 182.67

    x1 = torch.from_numpy(np.array([upper_arm], dtype='float32'))
    x1.view((1,1))        
    x, pred_LL, pred_UL, pred_LA, pred_UA = model(x1/389.098, flag)
    mae += abs(pred_UA - humerus/10)
    mse += (pred_UA - humerus/10)**2
    
mae/=index+1
mse/=index+1


print('Mean Absolute Error on ', index, 'samples :', mae.item())
print('Mean Squared Error on ', index, 'samples :', np.sqrt(mse.item()))





