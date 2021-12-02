from src.config import Config
from src.train import train
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='conf.yml',help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    mode = config.MODE
    if mode == 1:
        train(config)
    else:
        print('Not implemented yet')

if __name__ == "__main__":
    main()