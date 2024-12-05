import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss
        
        Args:
            pred (torch.Tensor): Predicted segmentation map
            target (torch.Tensor): Ground truth segmentation map
            
        Returns:
            torch.Tensor: Dice loss value
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class IoULoss(nn.Module):
    """
    IoU Loss for segmentation tasks
    """
    def __init__(self, smooth: float = 1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU Loss
        
        Args:
            pred (torch.Tensor): Predicted segmentation map
            target (torch.Tensor): Ground truth segmentation map
            
        Returns:
            torch.Tensor: IoU loss value
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target - pred * target).sum(dim=(2, 3))
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    """
    Combined Loss: Cross Entropy + Dice Loss
    """
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Combined Loss
        
        Args:
            pred (torch.Tensor): Predicted segmentation map
            target (torch.Tensor): Ground truth segmentation map
            
        Returns:
            torch.Tensor: Combined loss value
        """
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss