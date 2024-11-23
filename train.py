
from ultralytics import YOLO
from loguru import logger
import time
import argparse


data = 'coco_pcb.yaml'

root_path = 'lPCB'


epochs = 400
batch = 256

device = [0]


def train():
    project_name = 'LPCB'

    yaml_file = 'mobilenet/Replenishment/Neck/yolov8s-RepNCSPELAN4-CoT-SCFGhostv2.yaml'
    model = YOLO(yaml_file)

    model.train(data=data, cfg='default.yaml',
                device=device, name=project_name,
                batch=batch, epochs=epochs)


def main():
    train()




if __name__ == '__main__':

    main()
