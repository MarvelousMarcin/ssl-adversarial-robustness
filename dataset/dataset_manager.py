import os
import tarfile
import numpy as np
from torch.utils.data import DataLoader, Subset, IterableDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import urllib.request
import zipfile

class DatasetManager():
    def __init__(self, root="./data", batch_size=64, num_workers=4):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_cars(self, subset_size=None, split="train"):
        cars_dir = os.path.join(self.root, "CARS_196")
        os.makedirs(cars_dir, exist_ok=True)
        dataset = ImageFolder(root=os.path.join(cars_dir, split), transform=self.transform)
        if subset_size and subset_size < len(dataset):
            indices = np.random.RandomState(42).choice(len(dataset), subset_size, replace=False)
            indices.sort()
            dataset = Subset(dataset, indices.tolist())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def get_imagenet(self, subset_size=None, split="train"):
        imagenet_dir = os.path.join(self.root, "imagenette2")
        os.makedirs(imagenet_dir, exist_ok=True)
        dataset = ImageFolder(root=os.path.join(imagenet_dir, split), transform=self.transform)
        if subset_size and subset_size < len(dataset):
            indices = np.random.RandomState(42).choice(len(dataset), subset_size, replace=False)
            indices.sort()
            dataset = Subset(dataset, indices.tolist())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def get_cifar100(self, subset_size=None, split="train"):
        cifar100_dir = os.path.join(self.root, "cifar100")
        os.makedirs(cifar100_dir, exist_ok=True)
        dataset = ImageFolder(root=os.path.join(cifar100_dir, split), transform=self.transform)
        if subset_size and subset_size < len(dataset):
            indices = np.random.RandomState(42).choice(len(dataset), subset_size, replace=False)
            indices.sort()
            dataset = Subset(dataset, indices.tolist())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def get_cifar10(self, subset_size=None, split="train"):
        cifar10_dir = os.path.join(self.root, "cifar10")
        os.makedirs(cifar10_dir, exist_ok=True)
        dataset = ImageFolder(root=os.path.join(cifar10_dir, split), transform=self.transform)
        if subset_size and subset_size < len(dataset):
            indices = np.random.RandomState(42).choice(len(dataset), subset_size, replace=False)
            indices.sort()
            dataset = Subset(dataset, indices.tolist())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def get_sop(self, subset_size=None):
        sop_dir = os.path.join(self.root, "SOP")
        os.makedirs(sop_dir, exist_ok=True)

        url = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
        zip_path = os.path.join(sop_dir, "SOP.zip")
        extracted_path = os.path.join(sop_dir, "Stanford_Online_Products")

        if not os.path.exists(zip_path):
            print("Downloading SOP dataset from FTP...")
            urllib.request.urlretrieve(url, zip_path)

        if not os.path.exists(extracted_path):
            print("Extracting SOP dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(sop_dir)

        images_path = os.path.join(extracted_path, "images")
        dataset = ImageFolder(root=images_path, transform=self.transform)
        if subset_size and subset_size < len(dataset):
            indices = np.random.RandomState(42).choice(len(dataset), subset_size, replace=False)
            indices.sort()
            dataset = Subset(dataset, indices.tolist())

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
