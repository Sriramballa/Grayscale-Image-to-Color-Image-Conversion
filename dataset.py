import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ColorizationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)

        self.transform_gray = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.transform_color = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        gray = self.transform_gray(img)
        color = self.transform_color(img)

        return gray, color
