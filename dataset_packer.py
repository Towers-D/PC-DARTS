import torch
import torchvision.transforms as transforms
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train = False, transform = None):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        
        if train:
            self.mean = torch.mean(self.x, [0, 2, 3])
            self.std = torch.std(self.x, [0, 2, 3])
            self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        im = self.x[idx]
        
        if self.transform is not None:
            im = self.transform(im)
            
        return im, self.y[idx]
    
def pack_dataset(name:str, data_path):
    path = f'{data_path}/{name}/'
    train_x, train_y, valid_x, valid_y, test_x, test_y = (np.load(f'{path}{x}.npy') for x in ['train_x', 'train_y', 'valid_x', 'valid_y', 'test_x', 'test_y'])
    classes = len(np.unique(test_y))
    
    train_ds = Dataset(train_x, train_y, train = True)
    valid_ds = Dataset(valid_x, valid_y, transform = train_ds.transform)
    test_ds = Dataset(test_x, test_y, transform = train_ds.transform)
    
    return train_ds, valid_ds, test_ds, classes