import os

from .src.trainer import Trainer

def train(config):
    trainer = Trainer(config)
    trainer.train()
    print('training complete')