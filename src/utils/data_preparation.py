import torchvision
import torch
import albumentations as A
import numpy as np

class Transforms:
    def __init__(self, transforms: A.Compose):
        """
        Initialize a Transforms object.

        Parameters:
        transforms (A.Compose): An Albumentations composition of image transformations.

        Returns:
        None
        """
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        """
        Apply the defined image transformations to the input image.

        Parameters:
        img: Input image to apply transformations to.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

        Returns:
        np.ndarray: Transformed image as a NumPy array.
        """
        return self.transforms(image=np.array(img).astype(np.uint8))["image"]

class DataPreparation:
    def __init__(self, batch_size, train_transform, test_transform):
        """
        Initialize a DataPreparation object.

        Parameters:
        batch_size (int): Batch size for loading data.
        train_transform (A.Compose): Transformation to apply to training data.
        test_transform (A.Compose): Transformation to apply to test data.

        Returns:
        None
        """
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size

    def load_data(self, mode = "train"):
        """
        Load and prepare train and test data sets using the specified transforms. Returns both sets if the mode is train, returns only test if the mode is test
        
        Parameters:
        mode (string): Dataloading mode.

        Returns:
        torch.utils.data.DataLoader: DataLoader for the training set.
        torch.utils.data.DataLoader: DataLoader for the test set.
        """
        # Load train and test sets
        if mode == "train":
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=self.train_transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=self.test_transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            
        else:
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=self.test_transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            return testloader
        
        return trainloader, testloader
