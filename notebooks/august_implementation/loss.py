"""
The loss used is the Dice Loss
For reference, see etc.: 
https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b
"""

import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        """
        Returns 1.0 - Dice coefficient = loss
        """
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


# def diceloss(y_pred, y_true, smooth=1.0):
#     """
#     Returns 1.0 - Dice coefficient = loss
#     """
#     assert y_pred.size() == y_true.size()
    
#     ## Convert to 1D vector
#     y_pred = y_pred[:, 0].contiguous().view(-1)
#     y_true = y_true[:, 0].contiguous().view(-1)
    
#     ## Find intersection
#     intersection = (y_pred * y_true).sum()
    
#     ## Compute Dice coefficient
#     dsc = (2. * intersection + smooth) / (
#         y_pred.sum() + y_true.sum() + smooth)
#     return 1. - dsc