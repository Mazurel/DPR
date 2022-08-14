#!/usr/bin/env python3
'''
Virtual DPR wrapper to be used in size specific DPR's wrappers.
Based on provided test code.
Original repo: https://github.com/zhhoper/DPR.
'''

import os
import numpy as np
from random import randint

from torch.autograd import Variable
import torch
import cv2

class DPRBase:
    '''
    Wrapper class for simple DPR usage in form of relighting images !
    '''

    # This folder contains avaiable lights
    myPath = os.path.dirname(os.path.abspath(__file__))
    lightsFolder = os.path.join(myPath, "lights/")
    modelFolder = os.path.join(myPath, 'trained_model/')

    # Public methods
    def __init__(self):
        self.lights = sorted(map(lambda f: os.path.join(self.lightsFolder, f), os.listdir(self.lightsFolder)))

    def lightsAmount(self) -> int:
        return len(self.lights)

    def relighten(self, imagePath: str, light: int):
        raise NotImplementedError()

    def random_relighten(self, imagePath: str):
        return self.relighten(imagePath, randint(0, self.lightsAmount() - 1))

    def append_light(self, lightPath: str):
        self.lights.append(lightPath)

    # Private helper methods
    @staticmethod
    def _loadModel():
        '''
        Load pretrained DPR model from some path.
        '''
        raise NotImplementedError()

    @staticmethod
    def _loadLightFromFile(path):
        '''
        Loads light data from specified, newline separated, file
        '''
        sh = np.loadtxt(path)
        sh = sh[0:9]
        return sh * 0.7 # Scale down

    @staticmethod
    def _light2Input(light):
        '''
        Converts matrix representing light, into NN input matrix.
        '''
        sh = np.reshape(light, (1, 9, 1, 1)).astype(np.float32)
        return Variable(torch.from_numpy(sh).cuda())

    @staticmethod
    def _loadLabImage(path, size):
        '''
        Loads image in LAB format (ready to be transformed)
        '''
        img = cv2.imread(path)
        row, col, _ = img.shape
        img = cv2.resize(img, size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    @staticmethod
    def _lab2Input(labImage):
        '''
        Converts image in LAB format into Pytorch' input
        '''
        inputL = labImage[:, :, 0]
        inputL = inputL.astype(np.float32)/255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]
        return Variable(torch.from_numpy(inputL).cuda())

    @staticmethod
    def _output2Lab(inpt, output):
        '''
        Takes output from NN and its input and returns modeified input image
        '''
        inptCopy = inpt.copy()
        output = output[0].cpu().data.numpy()
        output = output.transpose((1,2,0))
        output = np.squeeze(output)
        inptCopy[:, :, 0] = (output*255.0).astype(np.uint8)
        return inptCopy

