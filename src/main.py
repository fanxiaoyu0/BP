import os
import pickle
import numpy as np
import random
from tqdm import tqdm


import torch
from torch import nn
from torch.utils import data

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)

class Dataset(data.Dataset):
    def __init__(self, _dataTensor):
        self._dataTensor = _dataTensor
        self.window_size = 16
    def __len__(self):
        return self._dataTensor.shape[0]-self.window_size
    def __getitem__(self, index):
        

        block=self._dataTensor[index:index+self.window_size,:]
        # print(block)
        data=torch.zeros((self.window_size,4),dtype=torch.float32).to('cuda')
        address_list=block[:,0]
        direction_list=block[:,1]
        current_address=self._dataTensor[index+self.window_size,0]
        current_direction=self._dataTensor[index+self.window_size,1]
        # print(address_list)
        # print(direction_list)
        # print(current_address)
        # print(current_direction)
        # print(torch.mul(address_list==current_address,direction_list==1))
        data[:,0]=(torch.mul(address_list==current_address,direction_list==1)).float()
        data[:,1]=(torch.mul(address_list==current_address,direction_list==0)).float()
        data[:,2]=(torch.mul(address_list!=current_address,direction_list==1)).float()
        data[:,3]=(torch.mul(address_list!=current_address,direction_list==0)).float()
        # data = self._dataTensor[index:index+self.window_size,1:2]
        # label=data[-1][0].to(torch.long)
        # data[-1][0]=-1
        # print(data)
        # print(label)
        # fsdhk
        local_window_size=16
        local_history_list=torch.zeros(local_window_size,2,dtype=torch.float32).to('cuda')
        # print(self._dataTensor[max(index+self.window_size-600,0):index+self.window_size,0])
        # print(self._dataTensor[max(index+self.window_size-600,0):index+self.window_size,0]==current_address)
        recent_history=self._dataTensor[max(index+self.window_size-1024,0):index+self.window_size]
        index=torch.nonzero(recent_history[:,0]==current_address).squeeze()
        # print(index)
        # local_history_list[:min(index.shape[0],16)]
        temp=torch.index_select(recent_history,0,index)[:,1]
        # print(temp)
        if temp.shape[0]>=local_window_size:
            local_history_list[:,0]=temp[-local_window_size:]
            local_history_list[:,1]=(index[-local_window_size:]/1024)
        elif temp.shape[0]>0:
            local_history_list[-temp.shape[0]:,0]=temp
            local_history_list[-temp.shape[0]:,1]=(index/1024)
        # local_history_list[:,1]=(1024-local_history_list[:,1])/1024
        
        # print(local_history_list)
        # print(temp)
        # fsdhjk
        # if index.shape[0]>16:
            # temp=temp[:16]
        # else:
            # temp=torch.cat((temp,torch.zeros(16-temp.shape[0],2,dtype=torch.long)),0)
        # p=0
        # for i in range(index+self.window_size):
        #     t=index+self.window_size-1-i
        #     if self._dataTensor[t][0]==current_address:
        #         local_history_list[p][0]=self._dataTensor[t][1]
        #         local_history_list[p][1]=i
        #         p+=1
        #         if p==16:
        #             break
        #     if i>600:
        #         break
        # print(local_history_list)
        return data,local_history_list,current_direction.to(torch.long)

def process_raw_data():
    history_list=[]
    with open('../data/LONG-SPEC2K6-00.res','r') as f:
        # for index,line in enumerate(tqdm(f.readlines())):
        for line in tqdm(f.readlines()):
            line=line.strip()
            if line=='':
                continue
            pc=int(line.split(' ')[0])
            direction=int(line.split(' ')[1])
            history_list.append(np.array([pc,direction],dtype=np.float32))
    # print(history_list[:10])
    history_list=np.array(history_list,dtype=np.float32)
    history_tensor=torch.from_numpy(history_list)
    # print(history_tensor.shape)
    pickle.dump(history_tensor,open('../data/history_tensor.pkl','wb'))

def construct_dataset(batchSize):
    history_tensor=pickle.load(open('../data/history_tensor.pkl','rb'))
    history_tensor=history_tensor.to('cuda')
    # print("to cuda")
    train_length=int(len(history_tensor)*0.8)
    train_dataset=Dataset(history_tensor[:train_length])
    test_dataset=Dataset(history_tensor[train_length:])
    # train_loader=data.DataLoader(train_dataset,batch_size=batchSize,shuffle=False)
    train_loader=data.DataLoader(train_dataset,batch_size=batchSize,shuffle=True)
    test_loader=data.DataLoader(test_dataset,batch_size=batchSize,shuffle=True)
    return train_loader,test_loader

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
#         self.fc = nn.Sequential(nn.Linear(16, 16),nn.ReLU(),nn.Linear(16, 2))
#         self.fc_1 = nn.Sequential(nn.Linear(16*4, 32),nn.ReLU(),nn.Linear(32, 16),nn.ReLU(),nn.Linear(16, 2))
#         self.fc_2 = nn.Sequential(nn.Linear(32*2, 256),nn.ReLU(),nn.Linear(256, 32),nn.ReLU(),nn.Linear(32, 2))
#     def forward(self, x1,x2):
#         # x1=self.fc_1(x1.view(x1.shape[0],-1))
#         # x1,_ = self.lstm(x1)
#         x2 = self.fc_2(x2.view(x2.shape[0],-1))
#         # x = torch.cat((x1[:, -1, :],x2),1)
#         # x = self.fc(x)
#         # x2 = self.fc_2(x2.view(x2.shape[0],-1))
#         # x=x2[:, -1, :]
#         # x=torch.concat((x1[:, -1, :], x2[:, -1, :]), 1)
#         # print(x.shape)
#         # x = self.fc(x)
#         x=x2
#         x = torch.softmax(x, dim=1)
#         return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc = nn.Sequential(nn.Linear(160, 64),nn.Dropout(0.2),nn.ReLU(),nn.Linear(64, 16),nn.Dropout(0.2),nn.ReLU(),nn.Linear(16, 2))
        # self.fc = nn.Sequential(nn.Linear(160, 160),nn.Sigmoid(),nn.Linear(16, 2))
        self.fc = nn.Sequential(nn.Linear(32, 1))
    def forward(self, x1):
        x1=self.fc(x1.view(x1.shape[0],-1))
        # x1,_ = self.lstm(x1)
        # x2 = self.fc_2(x2.view(x2.shape[0],-1))
        # x = torch.cat((x1[:, -1, :],x2),1)
        # x = self.fc(x)
        # x2 = self.fc_2(x2.view(x2.shape[0],-1))
        # x=x2[:, -1, :]
        # x=torch.concat((x1[:, -1, :], x2[:, -1, :]), 1)
        # print(x.shape)
        # x = self.fc(x1)
        x=x1
        # x = torch.softmax(x, dim=1)
        return x

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_1 = nn.LSTM(input_size=4, hidden_size=4, num_layers=1, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(8, 8),nn.ReLU(),nn.Linear(8, 2))
    def forward(self, x1,x2):
        x1,_ = self.lstm_1(x1)
        x2,_ = self.lstm_2(x2)
        # x=x2[:, -1, :]
        x=torch.concat((x1[:, -1, :], x2[:, -1, :]), 1)
        # print(x.shape)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=4, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(4, 4),nn.ReLU(),nn.Linear(4, 2))
    def forward(self, x):
        x,_ = self.gru(x)
        x = self.fc(x[:, -1, :])
        x = torch.softmax(x, dim=1)
        return x

def train_one_epoch(model:nn.Module, train_loader:data.DataLoader, criterion, optimizer, scheduler):
    model.train()
    right_count=0
    total_count=0
    losses=torch.zeros(len(train_loader),dtype=torch.float)
    # for index, (input, label) in enumerate(train_loader):
    for index,(input, local_history_list, label) in enumerate(tqdm(train_loader)):
        output = model(input, local_history_list)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[index]=loss

        predict_label = torch.argmax(output, axis=1)
        step_right_count=torch.sum(predict_label==label)
        right_count+=step_right_count
        total_count+=label.shape[0]
        if index%100==0:
            scheduler.step()
        print("loss:",loss.item(),"accuracy:",(right_count/total_count).item(),"step_accuracy:",(step_right_count/label.shape[0]).item())
    
    accuracy=(right_count/total_count).item()
    return (torch.sum(losses)/losses.nelement()).item(),accuracy

def test_one_epoch(model:nn.Module, test_loader:data.DataLoader):
    model.eval()
    right_count=0
    total_count=0
    # for index, (input, label) in enumerate(test_loader):
    for index,(input, label) in enumerate(tqdm(test_loader)):
        output = model(input)
        predict_label = torch.argmax(output, axis=1)
        right_count+=torch.sum(predict_label==label)
        total_count+=label.shape[0]
        if index%100==0:
            print("accuracy:",(right_count/total_count).item())
    accuracy=(right_count/total_count).item()
    return accuracy

def trial(model:nn.Module,modelName,epochs,batchSize,savedName):
    train_loader,test_loader=construct_dataset(batchSize=batchSize)
    model.to('cuda')
    # summary(model, (512, 100, 32))
    
    criterion=nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=15,eta_min=1e-6)

    maxBearableEpochs=30
    current_best_accuracy=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainLoss,trainAccuracy=train_one_epoch(model=model, train_loader=train_loader, criterion=criterion,optimizer=optimizer,scheduler=scheduler)
        test_accuracy=test_one_epoch(model=model, test_loader=test_loader)
        print("epoch:",epoch,"trainLoss:",trainLoss,"trainAccuracy:",trainAccuracy,"test_accuracy",test_accuracy,"current_best_accuracy",current_best_accuracy) #"delta:",validateScore-current_best_accuracy
        # writer.add_scalars(modelName,{'trainScore':trainScore,'validateScore':validateScore,"trainLoss":trainLoss,}, epoch)
        if sum(test_accuracy) > current_best_accuracy:
            current_best_accuracy=sum(test_accuracy)
            currentBestEpoch=epoch
            torch.save(model, "../weight/"+modelName+"/"+savedName)
        else:
            if epoch>=currentBestEpoch+maxBearableEpochs:
                break

def main(model:nn.Module,batchSize):
    model.to('cuda')
    # summary(model, (512, 100, 32))
    
    criterion=nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=15,eta_min=1e-6)

    pht=[0 for i in range(2**17)]
    ghr=0
    local_history_length=16
    max_ghr=2**17-1
    total_count=0
    right_count=0
    batch_data=torch.zeros((batchSize,local_history_length,1),dtype=torch.float)
    temp=[]
    label=[]
    count=0
    with open('../data/LONG-SPEC2K6-00.res','r') as f:
        index=0
        for line in tqdm(f.readlines()):
        # for line in tqdm(f.readlines()):
            line=line.strip()
            if line=='':
                continue
            index+=1
            pc=int(line.split(' ')[0])
            direction=int(line.split(' ')[1])
            label.append(direction)
            pht_index=(ghr^pc)%(2**17)
            pht_history=pht[pht_index]
            temp.append(pht_history)
            pht[pht_index]=((pht_history<<1)+direction)&(2**local_history_length-1)
            ghr=((ghr<<1)+direction)&max_ghr
            if (index)%batchSize==0:
                count+=1
                for i in range(batchSize):
                    a=torch.zeros((local_history_length,1),dtype=torch.float)
                    for j in range(local_history_length):
                        # print("(1<<j)",(1<<j))
                        # print("(temp[i]>>j)&1",(temp[i]>>j)&1)
                        a[j][0]=(temp[i]>>j)&1
                    batch_data[i]=a
                    # fsdhk
                batch_data=batch_data.to('cuda')
                label=torch.tensor(label,dtype=torch.long).to('cuda')
                # print(batch_data)
                # print(label)

                output=model(batch_data)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predict_label = torch.argmax(output, axis=1)
                step_right_count=torch.sum(predict_label==label)
                right_count+=step_right_count
                total_count+=label.shape[0]
                if count%1000==0:
                    scheduler.step()
                    print("loss:",loss.item(),"accuracy:",(right_count/total_count).item(),"step_accuracy:",(step_right_count/label.shape[0]).item())
                temp=[]
                label=[]

def sub(batchSize):
    model_list=[]
    for i in range(1024):
        model_list.append(MLP().to('cuda'))
    # model.to('cuda')
    # summary(model, (512, 100, 32))
    
    

    # pht=[0 for i in range(2**17)]
    ghr=0
    global_history_length=32
    # local_history_length=16
    # max_ghr=2**17-1
    total_count=0
    right_count=0
    ghr_batch=[]
    label_batch=[]
    with open('../data/LONG-SPEC2K6-00.res','r') as f:
        index=0
        for line in tqdm(f.readlines()):
            line=line.strip()
            if line=='':
                continue
            index+=1
            pc=int(line.split(' ')[0])
            direction=int(line.split(' ')[1])
            model=model_list[pc%1024]
            # criterion=nn.CrossEntropyLoss()
            criterion=nn.BCELoss()
            # criterion=nn.L1Loss()
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
            # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=15,eta_min=1e-6)
            
            ghr_batch.append(ghr)
            label_batch.append(direction)
            ghr=((ghr<<1)+direction)%(2**global_history_length)
            if index%batchSize==0:
                batch_data=torch.zeros((batchSize,global_history_length,1),dtype=torch.float)
                for i in range(batchSize):
                    a=torch.zeros((global_history_length,1),dtype=torch.float)
                    for j in range(global_history_length):
                        if ((ghr_batch[i]>>j)&1)==1:
                            a[j][0]=1
                        else:
                            a[j][0]=-1
                        # a[j][0]=(ghr_batch[i]>>j)&1
                    batch_data[i]=a
                batch_data=batch_data.to('cuda')
                label_batch=torch.tensor(label_batch,dtype=torch.float).unsqueeze(1).to('cuda')

                output=model(batch_data)
                loss = criterion(output, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predict_label = torch.argmax(output, axis=1)
                step_right_count=torch.sum(predict_label==label_batch)
                right_count+=step_right_count
                total_count+=label_batch.shape[0]
                if index%(1000*batchSize)==0:
                    # scheduler.step()
                    print("loss:",loss.item(),"accuracy:",(right_count/total_count).item(),"step_accuracy:",(step_right_count/label_batch.shape[0]).item())
                ghr_batch=[]
                label_batch=[]

if __name__ == '__main__':
    # process_raw_data()
    # model=LSTM()
    # trial(model=model,modelName="LSTM",epochs=200,batchSize=512,savedName="1.pth")

    # model=GRU()
    # trial(model=model,modelName="GRU",epochs=200,batchSize=512,savedName="1.pth")

    # model=MLP()
    # trial(model=model,modelName="MLP",epochs=200,batchSize=512,savedName="1.pth")

    # model=MLP()
    # main(model=model,batchSize=16)

    model=MLP()
    sub(batchSize=1)
    print("All is well!")