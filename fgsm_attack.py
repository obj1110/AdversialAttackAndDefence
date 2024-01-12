import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


## 我在想这个算法本身是不是也应该被修改一下，
class FGSM_ATTACK():

    def __init__(self,image, epsilon, data_grad):
        self.image = image
        self.epsilon = epsilon
        self.data_grad = data_grad

    def fgsm_attack(self):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = self.data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = self.image + self.epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image