import os
import h5py
from data.lmdb_datasets import LMDBDataset
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip as Flip


class Shapes3D(Dataset):
    """ 3D-Shapes dataset. """

    def __init__(self, root, train=True, transform=None):
        datapath = os.path.join(root, "3DShapes/3dshapes.h5")
        assert os.path.exists(datapath), "You need to download the data first!"
        data = h5py.File(datapath, 'r')
        self.images = data['images'][:]  # convert to tensor and place to RAM
        self.labels = data['labels'][:]  # convert to tensor and place to RAM
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_3dshapes(root, transform):
    # TODO: Watch out here, we do not have any splits
    trainset = Shapes3D(root=root, train=True, transform=transform)
    # validset = Shapes3D(root=root, train=False, transform=transform)
    return trainset, trainset


def get_cifar(root, transform, load_trainset=True):
    trainset = None
    if load_trainset:
        trainset = CIFAR10(root=root, download=True, train=True, transform=Compose([Flip(), transform]))
    validset = CIFAR10(root=root, download=True, train=False, transform=transform)
    return trainset, validset


def get_celebahq(root, transform, load_trainset=True):
    datapath = os.path.join(root, "CelebAHQ256")
    assert os.path.exists(datapath), "You need to download the data first!"
    trainset = None
    if load_trainset:
        trainset = LMDBDataset(root=datapath, name='celeba', train=True, transform=Compose([Flip(), transform]))
    validset = LMDBDataset(root=datapath, name='celeba', train=False, transform=transform)
    return trainset, validset


def get_imagenet32(root, transform, load_trainset=True):
    datapath = os.path.join(root, "ImageNet32")
    assert os.path.exists(datapath), "You need to download the data first!"
    trainset = None
    if load_trainset:
        trainset = LMDBDataset(root=datapath, name='imagenet-oord', train=True, transform=transform)
    validset = LMDBDataset(root=datapath, name='imagenet-oord', train=False, transform=transform)
    return trainset, validset


def get_dataset_specifications(dataset):
    assert dataset in ["CIFAR10", "ImageNet32", "CelebAHQ256", "3DShapes"]
    num_channels = image_size = None
    if dataset == "CIFAR10":
        num_channels, image_size = 3, 32
    elif dataset == "ImageNet32":
        num_channels, image_size = 3, 32
    elif dataset == "CelebAHQ256":
        num_channels, image_size = 3, 256
    elif dataset == "3DShapes":
        num_channels, image_size = 3, 64
    print("image dimensions: ", num_channels, "x", image_size, "x", image_size)
    return num_channels, image_size


def get_dataset(root, dataset, transform, load_trainset=True, condition=None):
    assert dataset in ["CIFAR10", "ImageNet32", "CelebAHQ256", "3DShapes"]
    trainset = validset = None
    if dataset == "CIFAR10":
        trainset, validset = get_cifar(root, transform, load_trainset)
    elif dataset == "ImageNet32":
        trainset, validset = get_imagenet32(root, transform, load_trainset)
    elif dataset == "CelebAHQ256":
        trainset, validset = get_celebahq(root, transform, load_trainset)
    elif dataset == "3DShapes":
        trainset, validset = get_3dshapes(root, transform)
    # keep only one label if specified
    if condition is not None:
        # trainset
        idx = trainset.targets == condition
        trainset.data = trainset.data[idx]
        trainset.targets = trainset.targets[idx]
        # validset
        idx = validset.targets == condition
        validset.data = validset.data[idx]
        validset.targets = validset.targets[idx]
    return trainset, validset
