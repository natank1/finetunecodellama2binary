import os
# import trans_utils as ut0
from create_dataset import dataset_tensor
from torch.utils.data import DataLoader
import torch
from pca_analysis import PCA_object
from  numpy.random import random as rnd
from ae_model import  AE
import torch.nn as nn
from torch.nn.functional import normalize
batch_size = 8
lr = 1e-5         # learning rate
w_d = 1e-5        # weight decay
momentum = 0.7
epochs = 15
mal_train_por = 0.8
from enum import Enum
class TRANS_MANN(Enum):
    BENUGN = 0
    MALIC = 1

if __name__ == '__main__':
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print ("deice=",device)
    list_of_files = ["folder_name"+i for i in os.listdir("folder_name")]


    print(len(list_of_files) )
    train_t = dataset_tensor(list_of_files)
    train_loader = DataLoader(train_t, batch_size=8, shuffle=True)

    model = AE()
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    # criterion =    nn.CosineSimilarity( )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)
    model.train()

    for i_epoc in range(epochs):
        print("i_epoch=", i_epoc)

        running_loss = 0.0
        counter_a = 0
        counta=0
        for batch_idx, data in enumerate(train_loader):

            xx, label  = data
            if device:
                xx = xx.to(device)
            c=1

            sample,a0  = model(xx)
            # x2 = normalize(xx.squeeze(), p=2.0, dim=1)

            # loss = criterion(xx, sample)
            loss = criterion(a0, sample)

            optimizer.zero_grad()
            loss.backward()
            print (loss)
            optimizer.step()
            counta+=1
            # if counta>50:
            #     break
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print ("uu =",epoch_loss)
    path ='/home/ubuntu/models/my_model4.bin'
    torch.save(model.state_dict(), path)

    model = AE()

    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

    list_of_files = ["test_folder" + i for i in os.listdir("test_folder")]
    test_t = dataset_tensor(list_of_files)
    test_loader = DataLoader(test_t, batch_size=8, shuffle=False)
    #

    dic_aggregation ={}
    for batch_idx, data in enumerate(test_loader):
        xx, label = data
        xx= xx.to(device)

        score,_ =model.enc(xx)


        labels= [int(i.item()) for i in label]
        llabels =len(labels)
        scores =[score[l,:] for l in range(llabels) ]
        for jj in range(llabels):
            dic_aggregation.setdefault(labels[jj], [])
            dic_aggregation[labels[jj]].append(scores[jj])

    lrnds = rnd(len(dic_aggregation[TRANS_MANN.BENUGN.value]))
    train_list = []
    test_list = []
    for j, index in enumerate(dic_aggregation[TRANS_MANN.BENUGN.value]):
        if lrnds[j] < mal_train_por:
            train_list.append(index)
        else:
            test_list.append(index)

    pc_obj =PCA_object(torch.stack(train_list),ncomp=8)
    good_data =pc_obj.prject_dat(torch.stack(test_list))
    bad_data = pc_obj.prject_dat(torch.stack(dic_aggregation[TRANS_MANN.MALIC.value]))
    print ("good ",good_data.shape)
    print("bad ", bad_data.shape)

    pc_obj.pl_tensor(good_data[:,[3,1,2]],bad_data[:,[3,1,2]],'r','o','g','*')
    pc_obj.pl_tensor(good_data[:, [0, 2, 3]], bad_data[:, [0, 2, 3]], 'r', 'o', 'g', '*')
    pc_obj.pl_tensor(good_data[:, [0, 1, 2]], bad_data[:, [0, 1, 2]], 'r', 'o', 'g', '*')
    pc_obj.pl_tensor(good_data[:, [2, 0, 1]], bad_data[:, [2, 0, 1]], 'r', 'o', 'g', '*')

    pc_obj.pl_tensor(good_data[:, [3, 0, 1]], bad_data[:, [3,0, 1]], 'r', 'o', 'g', '*')
    pc_obj.pl_tensor(good_data[:, [0, 1, 3]], bad_data[:, [0, 1, 3]], 'r', 'o', 'g', '*')
    # pc_obj.pl_tensor(good_data[:, [2,0,1  ]], bad_data[:, [2, 0, 1]], 'r', 'o', 'g', '*')

    print ("Thanks you!")