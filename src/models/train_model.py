from U_NET_model import UNet
from validation.loss import diceloss

import torch
import numpy as np
import datetime


def train_model(model, optimizer, num_epochs, train_dataset, val_dataset, CUDA=False, SAVE_CHECKPOINTS=False):
    
    ## Seed training run
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## Init grad (???)
    x = torch.autograd.Variable(torch.FloatTensor(np.random.random((1, 1, 512, 512))))

    ## Save loss for plot
    training_loss=[]
    validation_loss=[]
    
    ## Status parameters
    global_steps=0

    ## Running epochs
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        ## Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        model.eval()

        ## For grid in validation set
        for inputs, targets in val_dataset:

            inputs=inputs.unsqueeze(0)
            targets=targets.unsqueeze(0)
            # Add channel
            if CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass to get output/logits
            y_hat = model(x)
            
            # Calculate Loss: softmax --> cross entropy loss
            # outputs shifts channel one place left
            loss = diceloss(y_pred=y_hat, y_true=targets)

            # Getting gradients w.r.t. parameters
            epoch_validation_loss += loss.cpu().detach().numpy()
            
        model.train()
        # For grid in traning set
        for inputs,targets in train_dataset:
            inputs=inputs.unsqueeze(0)
            targets=targets.unsqueeze(0)
            # Add channel
            if CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass to get output/logits
            y_hat = model(x)
            
            # Calculate Loss: softmax --> cross entropy loss
            # outputs shifts channel one place left
            loss = diceloss(y_pred=y_hat, y_true=targets)
            
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Update loss
            epoch_training_loss += loss.cpu().detach().numpy()
            
            # Step taken
            global_steps+=1
            
        ## Save every epoch
        if SAVE_CHECKPOINTS:
            model_name=f'./checkpoint_saves/{type(model).__name__}_checkpoint_epoch_{epoch}_{datetime.now()}'
            # Send dict to memory
            torch.save(model.state_dict(), model_name)
            
        # Early breaking if validationloss increases 3 times
        if len(validation_loss)>3:
            if (validation_loss[-1]>=validation_loss[-2]) and (validation_loss[-1]>=validation_loss[-3]) and (validation_loss[-1]>=validation_loss[-4]):
                break
        # Save loss for plot
        training_loss.append(epoch_training_loss)
        validation_loss.append(epoch_validation_loss)
        print(f"Epoch {epoch}\nTraning loss: {epoch_training_loss}\nValidation loss {epoch_validation_loss}")


if __name__ == '__main__':

    ## Initialize model
    UNET_model = UNet()

    ## Initialize optimizer
    learning_rate = 0.8e-3
    optimizer = torch.optim.Adam(UNET_model.parameters(), lr=learning_rate, weight_decay=4e-3)

    ## Get dataset
    train_dataset = 1
    val_dataset = 1

    train_model(model=UNET_model, 
                optimizer=optimizer,
                num_epochs=50, 
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                CUDA=False,
                SAVE_CHECKPOINTS=False)

    ## Save model
    model_name=f'./checkpoint_saves/{type(UNET_model).__name__}_{datetime.now()}'
    # Send dict to memory
    torch.save(UNET_model.state_dict(), model_name)
    
    