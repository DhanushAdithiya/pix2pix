from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
from torchvision.transforms import functional
import torch.nn as nn

class Data(Dataset):
    def __init__(self, path, target_size=None):
        self.filenames = glob(str(Path(path)/ "*"))
        self.target_size = target_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filname = self.filenames[index]
        image = Image.open(filname)
        image = functional.to_tensor(image)
        image_width = image.shape[2]

        real = image[:,:,: image_width // 2]
        condition = image[:, :, : image_width // 2 :]

        target_size = self.target_size
        if target_size:
            condition = nn.functional.interpolate(condition, target_size)
            real = nn.functional.interpolate(real, target_size)

        return real, condition
