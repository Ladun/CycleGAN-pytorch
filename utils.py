import os
import random
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob(os.path.join(data_dir, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob(os.path.join(data_dir, f"{mode}B") + "/*.*"))

    def __getitem__(self, index) :
        item_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            item_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            item_B = Image.open(self.files_B[index % len(self.files_B)])

        if self.transform != None:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return item_A, item_B

    def __len__(self) :
        return max(len(self.files_A), len(self.files_B))

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (self.epochs - self.decay_epochs)




def save_model(d_A, d_B, g_AB, g_BA, model_folder):
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(d_A.state_dict(), os.path.join(model_folder, "d_A.pt"))
    torch.save(d_B.state_dict(), os.path.join(model_folder, "d_B.pt"))
    torch.save(g_AB.state_dict(), os.path.join(model_folder, "g_AB.pt"))
    torch.save(g_BA.state_dict(), os.path.join(model_folder, "g_BA.pt"))