import torch
from utils.tools import get_cur_time, seed_everything
from utils.dataset import CocoDetection, CollateFn
from utils.trainer import Trainer
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor


class Arguments:

    def __init__(self):
        # model and processor
        self.model_name_or_path = "facebook/detr-resnet-50"

        # training
        self.epochs = 23
        self.batch_size = 4
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.weight_decay = 1e-4

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_optimizers(args, model):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


if __name__ == "__main__":
    args = Arguments()

    seed_everything()

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder='./data/balloon/train', processor=processor)
    val_dataset = CocoDetection(img_folder='./data/balloon/val', processor=processor, train=False)

    # id2label
    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    # dataset
    collate_fn = CollateFn(processor)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

    # loading model
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                   revision="no_timm",
                                                   num_labels=len(id2label),
                                                   ignore_mismatched_sizes=True)

    optimizer = configure_optimizers(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_dataloader), eta_min=0,
                                                           last_epoch=-1, verbose=False)

    # start train
    trainer = Trainer(args=args,
                      model=model,
                      processor=processor,
                      optimizer=optimizer,
                      scheduler=scheduler)
    trainer.train(train_data_loader=train_dataloader, test_data_loader=val_dataloader)
    trainer.save_model(get_cur_time() + "/detr-finetuned-balloon")
