import argparse
import os
import pydoc

import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser("retrieve pytorch model weights from pytorch lightning checkpoint")
    parser.add_argument("--config_path", type=str, default="./config/base_contrastive.yaml")
    parser.add_argument("--pl_checkpoint_path", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, default="./assets/")

    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isfile(args.pl_checkpoint_path)

    with open(args.config_path, encoding="utf-8") as _file:
        hparams = yaml.load(_file, Loader=yaml.SafeLoader)

    pl_model = pydoc.locate(hparams["model"]["pl_class"]).load_from_checkpoint(
        args.pl_checkpoint_path,
        hparams=hparams,
        map_location="cpu",
    )
    weights_path = os.path.join(
        args.weights_dir, f'{hparams["model"]["encoder_name"]}_{hparams["model"]["hidden_size"]}.pt'
    )
    torch.save(pl_model.model.backbone.state_dict(), weights_path)
    print(f"saved model weights to {weights_path}")


if __name__ == "__main__":
    main()
