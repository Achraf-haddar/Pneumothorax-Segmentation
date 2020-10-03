import os
import sys
import torch

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset

# training csv file path
TRAINING_CSV = "Dataset/stage_2_train.csv"

# training and test batch sizes
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

# number of epochs
EPOCHS = 10

# define the encoder for U-NET
ENCODER = "resnet18"

# We use imagenet pretrained weights for the encoder
ENCODER_WEIGHTS = "imagenet"

# train on gpu
DEVICE = "cuda"

def train(dataset, data_loader, model, criterion, optimizer):
    # put the model in train mode
    model.train()

    # calculate number of batches
    num_batches = int(len(dataset) / data_loader.batch_size) 
    
    # init tqdm to track progress
    tk0 = tqdm(data_loader, total=num_batches)

    # loop over all batches
    for d in tk0:
        # fetch input images and masks
        # from dataset batch
        inputs = d["image"]
        targets = d["mask"]

        # move images and masks to cpu/gpu device
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

        # zero grad the optimizer
        optimizer.zero_grad()

        # forward step of model
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, targets)

        # backward
        loss.backward()

        # step the optimizer
        optimizer.step()
    
    # close tqdm
    tk0.close()

def evaluate(dataset, data_loader, model):
    # put model in eval mode
    model.eval()
    # init final_loss to 0
    final_loss = 0
    # calculate number of batches and init tqdm
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    # we need no_grad context of torch this save memory
    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            output = model(inputs)
            loss = criterion(output, targets)
            # add loss to final loss
            final_loss += loss
    # close tqdm
    tk0.close()
    # return average loss over all batches 
    return final_loss / num_batches

if __name__ == "__main__":
    # read the training csv file 
    df = pd.read_csv(TRAINING_CSV)

    # split data into training and validation
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )

    # training and validation images list/arrays
    training_images = df_train.ImageId.values
    validation_images = df_valid.ImageId.values

    # fetch unet model from segmentation models with specified encoder architecture
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation="sigmoid"
    )

    # segmentation model provides you with a preprocessing function
    # that can be used for normalizing images 
    # normalization is only applied on images and not masks
    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    # send model to device
    model.to(DEVICE)
    
    #init training dataset 
    # transform is True for training data
    train_dataset = SIIMDataset(
        training_images,
        transform=True,
        preprocessing_fn=prep_fn
    )
    # wrap training dataset in torch's dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=12
    ) 
    
    #init validation dataset 
    # augmentations is disabled
    valid_dataset = SIIMDataset(
        validation_images,
        transform=False,
        preprocessing_fn=prep_fn
    )
    # wrap validation dataset in torch's dataloader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    ) 

    # criterion !!!!!
    criterion = nn.CrossEntropyLoss()
    
    # we will use Adam optimizer for faster convergence          
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # reduce learning rate when we reach a plateau on loss
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )
    
    # some logging
    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")
    print(f"Encoder: {ENCODER}")
    
    # loop over all epochs
    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch}")
        # train for one epoch
        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer
        )
        print(f"Validation Epoch: {epoch}")
        # calculate validation loss
        val_log = evaluate(
        valid_dataset,
        valid_loader,
        model
        )
        # step the scheduler
        scheduler.step(val_log["loss"])
        print("\n")