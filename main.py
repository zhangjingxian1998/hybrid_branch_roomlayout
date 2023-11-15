import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from models.detector import Detector
import random

def main(args):
    if args.mode == 'train':
        checkpoint_callback = ModelCheckpoint(monitor='train_loss',mode='min',save_top_k=3,save_last=True,
                                              filename='../save_model/{epoch:02d}-{train_loss:.5f}',
                                              )
        model = Detector()
        model.add_extra_args(args=args)
        trainer = pl.Trainer(
                    gpus=args.gpu_num,
                    check_val_every_n_epoch=1,
                    strategy='ddp',
                    sync_batchnorm = True,
                    max_epochs = args.epochs,
                    callbacks=[checkpoint_callback],
                    )
        trainer.fit(model)
    elif args.mode == 'test':
        trainer = pl.Trainer(
                    gpus=[int(args.test_gpu_num)],
                    enable_checkpointing=False,
                    # limit_test_batches=0.1,
                    )
        model = Detector.load_from_checkpoint(args.weight)
        model.add_extra_args(args=args)
        # model = model.load_from_checkpoint(args.weight)
        trainer.test(model)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='test') # train test
    parser.add_argument('--weight',default='')
    parser.add_argument('--data' ,default='Structured3D')
    parser.add_argument('--batch_size',default=16)
    parser.add_argument('--num_workers',default=12)
    parser.add_argument('--epochs',default=50)
    parser.add_argument('--test_gpu_num',default=0)
    parser.add_argument('--gpu_num',default=2)
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse()
    random.seed(123)
    torch.manual_seed(123)
    np.random.seed(123)
    torch.cuda.manual_seed(123)
    main(args)
