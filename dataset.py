import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data, self.labels = data, labels
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        image = np.expand_dims((self.data[index].reshape(28, 28)), axis=-1)
        label = None
        
        if self.labels is not None:
            label = self.labels[index]
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        image = np.expand_dims((self.data[index].reshape(28, 28)), axis=-1)
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image