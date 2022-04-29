import torch.nn as nn
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        """
        Returns 1.0 - Dice coefficient = loss
        """
        # print(y_pred)
        # print(y_true)
        assert y_pred.size() == y_true.size()
        
        ## Convert to 1D vector
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        
        ## Find intersection
        intersection = (y_pred * y_true).sum()
        
        ## Compute Dice coefficient
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc


def train_model(model, optimizer, num_epochs, train_dataset, val_dataset, CUDA=False, SAVE_CHECKPOINTS=False, size=512):
    
    ## Seed training run
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## Init grad (???)
    x = torch.autograd.Variable(torch.FloatTensor(np.random.random((1, 1, size, size))))

    ## Save loss for plot
    training_loss=[]
    validation_loss=[]
    
    ## Status parameters
    global_steps=0

    diceloss = DiceLoss()

    ## Running epochs
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        ## Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        model.eval()

        ## For grid in validation set
        for inputs, targets in tqdm(val_dataset):

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
        for inputs,targets in tqdm(train_dataset):
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
            
        # ## Save every epoch
        # if SAVE_CHECKPOINTS:
        #     model_name=f'/home/augustsemrau/drive/M1semester/02506_AdvancedImageAnalysis/02505miniproject/src/model/saved_models/{type(model).__name__}_checkpoint_epoch_{epoch}_size_{size}.pt'
        #     # Send dict to memory
        #     torch.save(model.state_dict(), model_name)
            
        # Save loss for plot
        training_loss.append(epoch_training_loss)
        validation_loss.append(epoch_validation_loss)
        model_name=f'/home/augustsemrau/drive/M1semester/02506_AdvancedImageAnalysis/02505miniproject/src/model/saved_models/{type(model).__name__}_checkpoint_epoch_{epoch}_size_{size}.pt'
        # Send dict to memory
        torch.save(model.state_dict(), model_name)

        # Early breaking if validationloss increases 3 times
        if len(validation_loss)>3:
            if (validation_loss[-1]>=validation_loss[-2]) and (validation_loss[-1]>=validation_loss[-3]) and (validation_loss[-1]>=validation_loss[-4]):
                break

        print(f"Traning loss: {epoch_training_loss}\nValidation loss {epoch_validation_loss}")
    
    model_name=f'/home/augustsemrau/drive/M1semester/02506_AdvancedImageAnalysis/02505miniproject/src/model/saved_models/{type(model).__name__}_checkpoint_epoch_last_size_{size}.pt'
    # Send dict to memory
    torch.save(model.state_dict(), model_name)
    return model, training_loss, validation_loss, global_steps



    




