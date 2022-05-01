import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
from tqdm import tqdm
from argparse import ArgumentParser

from src.data.datasets import DatasetPatch
from src.model.model_train import DiceLoss
from src.model.U_NET_model import UNet

def model_predict_validation_data(model_path, patch_size):
    '''
    Loads model from the given model path and predicts on validation images.

    :param model_path:      Path for model to load
    :param patch_size:      Size of patches to split 

    :returns:               Label images for segmentation produced by the model predictions
    '''

    with open(model_path, 'rb') as model_file:
        state_dict = torch.load(model_file)
        model = UNet(num_classes=1, in_channels=1, depth=4, merge_mode='concat')
        model.load_state_dict(state_dict)
        model.eval()

    
    validation_data = DatasetPatch(train=True, val=True, size=patch_size)

    predicted = []
    true = []

    print('Prediction on validation data...')
    for input, target in tqdm(validation_data):
        input = input.unsqueeze(0).float()
        target = target.unsqueeze(0).float()
        true.append(target)
        predicted.append(model(input))

    return list(zip(predicted, true))


if __name__ == '__main__':
    dice_loss = DiceLoss()

    parser = ArgumentParser()

    parser.add_argument('--model_name', '-m', type=str, required=True, help='Name of model e.g. UNet_checkpoint_epoch_0_size_512')
    parser.add_argument('--patch_size', '-p', type=int, required=True, help='''Size of patches for splitting images. 
                                                                        Should be a single number to define the quadratic
                                                                        patch of size (patch_size X patch_size)''')

    args = parser.parse_args()

    # Remove file ending if user adds .pt to model_name
    model_name = os.path.splitext(args.model_name)[0]
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models', model_name + '.pt')

    predictions_labels = model_predict_validation_data(model_path, args.patch_size)

    mean_loss = sum([dice_loss.forward(p, t).item() for (p,t) in predictions_labels])/len(predictions_labels)

    print(f'Model mean loss on validation data: {mean_loss}')