#!/usr/bin/env python3
'''
Simple DPR wrapper for generating relightened 1024x1024 images.
Based on provided test code.
Original repo: https://github.com/zhhoper/DPR.
'''

import os
import torch
import cv2

from transfrom_base import DPRBase

# For loading model
from model.defineHourglass_1024_gray_skip_matchFeature import *

class DPR_1024(DPRBase):
    '''
    Wrapper class for simple DPR usage in form of relighting (1024x1024) images !
    '''
    modelName = 'trained_model_1024_03.t7'
    size = (1024, 1024)

    # Public methods
    def __init__(self, target_size=None):
        super().__init__(target_size)
        self.network = DPR_1024._loadModel()

    # TODO: Consider returning LAB Color mapped images
    #       for optimization purposes
    def relighten(self, imagePath: str, light: int):
        raw_sh = DPRBase._loadLightFromFile(self.lights[light])
        sh = DPRBase._light2Input(raw_sh)
        labImage = DPRBase._loadLabImage(imagePath, DPR_1024.size)
        inputImage = DPRBase._lab2Input(labImage)
        noutputImg, _, _, _  = self.network(inputImage, sh, 0)
        outputImg = DPRBase._output2Lab(labImage, noutputImg)
        resultLab = cv2.cvtColor(outputImg, cv2.COLOR_LAB2RGB)
        resultImage = cv2.resize(resultLab, DPR_1024.size)
        return self.resize_to_target(cv2.cvtColor(labImage, cv2.COLOR_LAB2RGB)), self.resize_to_target(resultImage)

    # Private helper methods
    @staticmethod
    def _loadModel():
        '''
        Load pretrained DPR model from some path.
        '''
        network_512 = HourglassNet(16)
        network = HourglassNet_1024(network_512, 16)
        network.load_state_dict(torch.load(os.path.join(DPRBase.modelFolder, DPR_1024.modelName)))
        network.cuda()
        network.train(False)
        return network


# Simple example:

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dpr = DPR_1024()
    i, o = dpr.random_relighten("./data/obama.jpg")

    plt.figure("Before")
    plt.imshow(i)
    plt.figure("After")
    plt.imshow(o)
    plt.show()
