import torch
from tqdm import tqdm
from pathlib import Path
from utils.tools import get_lr
from accelerate import Accelerator


class Trainer:

    def __init__(self,
                 args = None,
                 model=None,
                 processor=None,
                 optimizer=None,
                 scheduler=None,
                 accelerator=None,
                 ):
        self.args = args
        if self.args is None:
            raise ValueError("args is None!")

        self.model = model
        self.model.to(self.args.device)

        self.processor = processor
        if self.processor is None:
            raise ValueError("tokenizer is None!")

        self.optimizer = optimizer
        if optimizer is None:
            raise ValueError("optimizer is None!")

        self.scheduler = scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()

    def train(self, train_data_loader=None, test_data_loader=None):
        for epoch in range(1, self.args.epochs + 1):
            train_total_loss = 0
            self.model.train()
            with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                      desc=f'Epoch: {epoch}/{self.args.epochs}',
                      postfix=dict) as train_pbar:
                for step, batch in train_pbar:
                    pixel_values = batch["pixel_values"].to(self.args.device)
                    pixel_mask = batch["pixel_mask"].to(self.args.device)
                    labels = [{k: v.to(self.args.device) for k, v in t.items()} for t in batch["labels"]]

                    # backward, calculate gradient
                    if self.accelerator is not None:
                        with self.accelerator.autocast():
                            # forward
                            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                            loss, loss_dict = outputs.loss, outputs.loss_dict
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    else:
                        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                        loss, loss_dict = outputs.loss, outputs.loss_dict
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()  # zero the gradient
                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    if self.accelerator is not None:
                        train_total_loss += self.accelerator.gather(loss).item()
                    else:
                        train_total_loss += loss.item()

                    train_pbar.set_postfix(
                        **{'train average loss': train_total_loss / (step + 1), 'train loss': loss.item(),
                           "lr": get_lr(self.optimizer)})
            # test
            if test_data_loader is not None:
                test_total_loss = 0
                with tqdm(enumerate(test_data_loader), total=len(test_data_loader),
                          desc=f'Epoch: {epoch}/{self.args.epochs}',
                          postfix=dict) as test_pbar:
                    self.model.eval()
                    for step, batch in test_pbar:
                        pixel_values = batch["pixel_values"].to(self.args.device)
                        pixel_mask = batch["pixel_mask"].to(self.args.device)
                        labels = [{k: v.to(self.args.device) for k, v in t.items()} for t in batch["labels"]]
                        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                        loss, loss_dict = outputs.loss, outputs.loss_dict

                        # tqdm
                        test_total_loss += loss.item()
                        test_pbar.set_postfix(
                            **{'test average loss': test_total_loss / (step + 1), 'test loss': loss.item()})

    def save_model(self, out_dir: str = None):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()
        self.model.save_pretrained(out_dir, torch_dtype=torch.float16)
        self.processor.save_pretrained(out_dir)
