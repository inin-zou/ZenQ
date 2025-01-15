import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from boson_sampler import BosonSampler
from utils import accuracy
import perceval as pcvl
import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway


import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from boson_sampler import BosonSampler
from utils import accuracy
import perceval as pcvl
import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway


class MnistModel(nn.Module):
    def __init__(self, device = 'cpu', embedding_size = 0):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        
        # 1st Convolutional layer (input: 1 channel, output: 32 channels, kernel size: 3x3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 2nd Convolutional layer (input: 32 channels, output: 64 channels, kernel size: 3x3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 3rd Convolutional layer (input: 64 channels, output: 128 channels, kernel size: 3x3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # MaxPooling layer to reduce the spatial dimension (2x2 pool)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer for classification after flattening
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)  # input size will change depending on the output of conv layers
        self.fc2 = nn.Linear(1024, 10)  # Output layer for 10 classes (MNIST)

    def forward(self, xb, emb=None):
        # Apply convolutional layers with ReLU activation and max pooling
        xb = self.pool(F.relu(self.conv1(xb)))
        xb = self.pool(F.relu(self.conv2(xb)))
        xb = self.pool(F.relu(self.conv3(xb)))
        
        # Flatten the output from the convolutional layers
        xb = xb.view(-1, 128 * 3 * 3)  # Flatten the 128 channels of size 3x3 (adjust as necessary)
        
        # If using embedding size, concatenate the embeddings with the flattened features
        if self.embedding_size and emb is not None:
            xb = torch.cat((xb, emb), dim=1)
        
        # Pass through fully connected layers
        xb = F.relu(self.fc1(xb))
        out = self.fc2(xb)
        return out

    def training_step(self, batch, emb=None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device))  # Generate predictions with embeddings
        else:
            out = self(images)  # Generate predictions without embeddings
        loss = F.cross_entropy(out, labels)  # Calculate the loss
        acc = accuracy(out, labels)
        return loss, acc

    def validation_step(self, batch, emb=None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device))  # Generate predictions with embeddings
        else:
            out = self(images)  # Generate predictions without embeddings
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        return result['val_loss'], result['val_acc']

# evaluation of the model
def evaluate(model, val_loader, bs: BosonSampler = None):
    if model.embedding_size:
        outputs = []
        for step, batch in enumerate(tqdm(val_loader)):
            # embedding in the BS
            images, labs = batch
            images = images.squeeze(0).squeeze(0)
            t_s = time.time()
            embs = bs.embed(images, 1000)
            outputs.append(model.validation_step(batch, emb=embs.unsqueeze(0)))
    else:
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
