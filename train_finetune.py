import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from torchvision.models import ResNet50_Weights

def main(hparams):
    img_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    # dm = TextImageDataModule.from_argparse_args(hparams, custom_tokenizer=tokenizer)
    # trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32)
    dm = TextImageDataModule(hparams.folder, hparams.batch_size, custom_tokenizer=tokenizer)
    trainer = Trainer(precision=16, max_epochs=32, strategy='ddp_find_unused_parameters_true')
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
