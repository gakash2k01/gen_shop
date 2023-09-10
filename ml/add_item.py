import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
import torch
import torchvision
from torchvision import transforms
import shutil
from csv import DictWriter
from tqdm import tqdm
import random
import numpy as np
import pickle

class Identity(nn.Module):
    """
    A simple identity layer used for network customization.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass through the identity layer.

        Parameters:
        - x (tensor): Input tensor.

        Returns:
        Tensor: The same input tensor without modification.
        """
        return x

class CatalogDataSet(Dataset):
    """
    Custom dataset class for loading catalog images.

    Parameters:
    - folder (str): Path to the folder containing image files.
    - transform (torchvision.transforms.Compose): Transformations to apply to the images.
    - augment_transform (torchvision.transforms.Compose): Augmentation transformations.

    Attributes:
    - folder (str): Path to the image folder.
    - transform (torchvision.transforms.Compose): Image transformations.
    - augment_trans (torchvision.transforms.Compose): Augmentation transformations.
    - items (list): List of items in the folder.
    - files (list): List of image file names.
    """
    def __init__(self, folder, transform=None, augment_transform=None):
        self.folder = folder
        self.transform = transform
        self.augment_trans = augment_transform
        self.items = os.listdir(self.folder)
        self.files = [item for item in self.items if os.path.isfile(os.path.join(self.folder, item))]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        int: Number of image files in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        tuple: A tuple containing the image tensor and the corresponding file name.
        """
        img_path = os.path.join(self.folder, self.files[idx])
        image = read_image(img_path)
        image = image.type(torch.FloatTensor)
        if self.transform:
            image = self.transform(image)
        if self.augment_trans:
            image = self.augment_transform(image)
        image = torch.mul(image, (1/255))
        return image, self.files[idx]

def make_data(model, cloth_class):
    """
    Generate embeddings for catalog items using a given model.

    Parameters:
    - model (nn.Module): The model to generate embeddings.
    - cloth_class (str): The class of clothing for the catalog.

    Returns:
    dict: A dictionary containing embeddings for each catalog item.
    """
    dataset = CatalogDataSet(f'zalando/{cloth_class}', transform)
    train_dataloader = DataLoader(dataset, batch_size=1024)
    embedding = {}
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for inputs, fn in tqdm(train_dataloader):
        inputs = inputs.to(device)
        inputs = inputs.type(torch.float)
        out = model(inputs)
        for idx in range(inputs.shape[0]):
            embedding[fn[idx][:-4]] = np.array(out[idx].cpu())
    return embedding

def append_data(record, cloth_class):
    """
    Appends item data to the catalog CSV file and copies the item image.

    Parameters:
    - record (str): The item identifier.
    - cloth_class (str): The class of clothing the item belongs to.

    Returns:
    None
    """
    # Create a unique item name combining the cloth class and record
    item = f"{cloth_class}_{record}"

    # Generate random original and discounted prices
    orig_price = random.randint(1000, 9999)
    disc_price = random.randint(100, 999)

    # Define the field names for the CSV
    field_names = ['id', 'name', 'orig_price', 'disc_price']

    # Create a dictionary entry for the item
    entry = {'id': record, 'name': item, 'orig_price': orig_price, 'disc_price': disc_price}

    # Open the CSV file in append mode and write the entry
    with open('catalog/data.csv', 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(entry)

    # Copy the item image to the products_images directory
    image_source_path = f"zalando/{cloth_class}/{record}.jpg"
    image_dest_path = f"catalog/products_images/{record}.jpg"
    shutil.copy(image_source_path, image_dest_path)

if __name__ == "__main__":
    """
    The main script for generating catalog item embeddings.

    This script performs the following steps:
    1. Initializes parameters and paths.
    2. Defines normalization and transformation for images.
    3. Sets up the model and prepares it for inference.
    4. Loops through clothing classes to generate embeddings.
    5. Appends item data to the catalog CSV file.
    6. Saves generated embeddings as a pickle file.
    """

    # List of clothing classes for processing
    cloth_classes = ['hoodies', 'hoodies-female', 'longsleeve', 'shirt', 'sweatshirt', 'sweatshirt-female']

    # Create directory for product images if it doesn't exist
    if not os.path.exists('./catalog/products_images'):
        os.mkdir('./catalog/products_images')

    # Define normalization and transformation for images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        normalize,
        transforms.Resize(255),
        transforms.CenterCrop(224)])

    # Set up the model and customize the fully connected layer
    model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
    model.fc = Identity()
    for params in model.parameters():
        params.requires_grad = False

    # Loop through clothing classes to generate embeddings
    for cloth_class in cloth_classes:
        # Generate embeddings for the current clothing class
        embeddings = make_data(model=model, cloth_class=cloth_class)

        # Append item data to the catalog CSV file and copy images
        for entry in embeddings:
            append_data(entry, cloth_class=cloth_class)

        # Save generated embeddings as a pickle file
        with open('catalog/embeds.pickle', 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
