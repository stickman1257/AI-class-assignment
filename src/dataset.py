import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as L
from PIL import Image
from typing import Literal

SEED = 36
L.seed_everything(SEED)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.transpose(image.reshape(3, 32, 32), (1, 2, 0))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.transform = transform
        self.classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.is_test = is_test
        
        self.samples = []
        if is_test:
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    for class_name in self.classes:
                        if class_name.lower() in img_name.lower():
                            label = self.class_to_idx[class_name]
                            self.samples.append((os.path.join(root_dir, img_name), label))
                            break
        else:
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    class_samples = [
                        (os.path.join(class_dir, img_name), self.class_to_idx[class_name])
                        for img_name in os.listdir(class_dir)
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    self.samples.extend(class_samples)
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        if is_test:
            print(f"Found {len(self.samples)} test images")
            class_counts = {}
            for _, label in self.samples:
                class_counts[label] = class_counts.get(label, 0) + 1
            print("Class distribution:", {self.classes[k]: v for k, v in class_counts.items()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.base_transform(image)
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomDataModule(L.LightningDataModule):
    def __init__(self, dataset_type: Literal['cifar10', 'flowers'] = 'cifar10', data_path: str = 'data', batch_size: int = 32):
        super().__init__()
        self.dataset_type = dataset_type
        self.data_path = str(data_path) if not isinstance(data_path, str) else data_path
        self.batch_size = batch_size
        self.cifar_mean = (0.4914, 0.4822, 0.4465)
        self.cifar_std = (0.2023, 0.1994, 0.2010)
        self.flowers_mean = (0.485, 0.456, 0.406)
        self.flowers_std = (0.229, 0.224, 0.225)
        self._setup_transforms()

    def _setup_transforms(self):
        if self.dataset_type == 'cifar10':
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_mean, self.cifar_std)
            ])
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_mean, self.cifar_std)
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Normalize(self.flowers_mean, self.flowers_std)
            ])
            self.transform = transforms.Compose([
                transforms.Normalize(self.flowers_mean, self.flowers_std)
            ])

    def setup(self, stage: str):
        if self.dataset_type == 'cifar10':
            self._setup_cifar10(stage)
        else:
            self._setup_flowers(stage)

    def _setup_cifar10(self, stage):
        if stage == 'fit' or stage == 'test':
            train_data = []
            train_labels = []
            for i in range(1, 6):
                batch = unpickle(os.path.join(self.data_path, 'cifar-10', f'data_batch_{i}'))
                train_data.append(batch[b'data'])
                train_labels.extend(batch[b'labels'])
            
            train_data = np.vstack(train_data)
            train_labels = np.array(train_labels)

            test_batch = unpickle(os.path.join(self.data_path, 'cifar-10', 'test_batch'))
            test_data = test_batch[b'data']
            test_labels = np.array(test_batch[b'labels'])

            val_size = int(len(train_data) * 0.2)
            train_data, val_data = train_data[:-val_size], train_data[-val_size:]
            train_labels, val_labels = train_labels[:-val_size], train_labels[-val_size:]

            if stage == 'fit':
                self.train_dataset = CIFAR10Dataset(train_data, train_labels, self.train_transform)
                self.val_dataset = CIFAR10Dataset(val_data, val_labels, self.transform)
            
            if stage == 'test':
                self.test_dataset = CIFAR10Dataset(test_data, test_labels, self.transform)

    def _setup_flowers(self, stage):
        if stage == 'fit' or stage == 'test':
            full_dataset = FlowerDataset(
                root_dir=os.path.join(self.data_path, 'archive/train'),
                transform=self.transform,
                is_test=False
            )
            
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size) 
            val_size = int(0.15 * total_size) 
            test_size = total_size - train_size - val_size  
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42) 
            )
            
            if stage == 'fit':
                self.train_dataset = train_dataset
                self.train_dataset.dataset.transform = self.train_transform
                self.val_dataset = val_dataset
            
            if stage == 'test':
                self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4) 