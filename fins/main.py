import os
import argparse
import torch
from torch.utils.data import DataLoader

from fins.data.process_data import load_rir_dataset, load_speech_dataset
from fins.trainer import Trainer
from fins.dataloader import ReverbDataset
from fins.model import FilteredNoiseShaper
from fins.utils.utils import load_config


def main(args):
    # load config
    config_path = "fins/config.yaml"
    config = load_config(config_path)
    print(config)

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"

    train_rir_list, valid_rir_list, test_rir_list = load_rir_dataset()
    train_source_list, valid_source_list, test_source_list = load_speech_dataset()

    # load dataset
    train_dataset = ReverbDataset(train_rir_list, train_source_list, config.dataset.params, use_noise=True)
    valid_dataset = ReverbDataset(valid_rir_list, valid_source_list, config.dataset.params, use_noise=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.train.params.num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.train.params.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.train.params.num_workers,
    )

    print("Number of RIR data", len(train_rir_list), len(valid_rir_list))
    print("Number of speech data", len(train_source_list), len(valid_source_list))
    print("Number of batches", len(train_dataloader), len(valid_dataloader))

    # load model
    model = FilteredNoiseShaper(config.model.params)

    # run trainer
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.train.params, config.eval.params, args)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("--save_name", type=str, default="m")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()

    main(args)
