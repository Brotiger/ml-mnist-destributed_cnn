
import torch.distributed as dist
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from transforms import data_tfs
from dataset import TrainDataset
from net import MnistNet
        
def train(gpu, gpu_count,max_epochs,input_dir):
    train_data = pd.read_csv("./dataset/train.csv")
    labels = train_data['label']
    train_data = train_data.drop(columns=["label"])
    train_data.head()
    
    accurancy = {"train": [], "valid": []}
    losses = {"train": [], "valid": []}

    rank = gpu
    word_size = gpu_count
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=word_size,                              
    	rank=rank                                               
    )  

    model = MnistNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    fLoss = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    train_dataset = TrainDataset(train_data.to_numpy().astype(np.float32), labels.to_numpy().astype(np.float32), transform=data_tfs)

    train_size = int(len(train_data) * 0.8)
    val_size = len(train_data) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=gpu_count,
    	rank=rank
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	val_dataset,
    	num_replicas=gpu_count,
    	rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=160,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=160,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=val_sampler
    )

    loaders = {"train": train_loader, "valid": val_loader}
    
    for epoch in range(max_epochs):
        if gpu == 0:
            print(f'Epoch: {epoch + 1}')

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
            
        for k, dataloader in loaders.items():
            epoch_all = 0
            epoch_currect = 0
            epoch_loss = []
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.cuda(non_blocking=True)
                y_batch = y_batch.type(torch.LongTensor).cuda(non_blocking=True)
                if k == "train":
                    model.train()
                    outp = model(x_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                
                loss = fLoss(outp, y_batch)
                if gpu == 0:
                    epoch_loss.append(loss.detach().item())
                
                if k == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if gpu == 0:
                    preds = outp.argmax(1)
                    epoch_currect += (preds.flatten() == y_batch).type(torch.float32).sum().item()
                    epoch_all += y_batch.shape[0]
                    
            if gpu == 0: 
                acc = epoch_currect/epoch_all
                loss = np.mean(epoch_loss)
                losses[k].append(loss)
                print(f'Loader: {k}, accuracy {acc}, loss: {loss}')