import os
from tqdm import tqdm

import torch
import torchvision
import numpy as np
import random
import cv2
import time

import logging

from .backbones import count_parameters

class FaceEmbeddingModel:

    def __init__(self, backbone, header, loss):
        self.backbone = backbone
        self.header = header
        self.loss = loss

        # join parameter sets of backbone and header
        self.params = list(backbone.parameters())
        self.params.extend(list(header.parameters()))


        self.params = [{'params': backbone.parameters()}, {'params': header.parameters()}]

        self.name = f"model_{backbone.__class__.__name__}-{header.__class__.__name__}-{loss.__class__.__name__}".lower()

        print(f"Built Embedding Model: [{backbone.__class__.__name__} -> {header.__class__.__name__} -> {loss.__class__.__name__}]")
        print(f"#Trainable parameters: {count_parameters(backbone)} (Backbone)")
        print(f"#Trainable parameters: {count_parameters(header)} (Header)")
        print(f"#Trainable parameters: {count_parameters(loss)} (Loss)")

        self.loss_window = 100


    def fit(self, train_dataset, batch_size, device, optimizer, max_epochs=0, max_training_steps=0, lr_global_step_scheduler=None, lr_epoch_scheduler=None, evaluator=None, val_dataset=None, evaluation_steps=0, tensorboard=False):
        assert max_epochs > 0 or max_training_steps > 0

        training_id = self.name + '_' + str(random.randint(0, 9999999)).zfill(7)
        training_path = os.path.join('output', training_id)
        if not os.path.exists(training_path):
            os.makedirs(training_path)

        logging.basicConfig(
            filename=os.path.join(training_path, 'training.log'),
            level=logging.INFO,
            format='[%(levelname)s] %(asctime)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p')
        print(f"Logs will be written to {os.path.join(training_path, 'training.log')}")

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if tensorboard:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(logdir=os.path.join('tensorboard', training_id))

            def evaluate(global_step):
                features, labels, thumbnails = self.encode_dataset(val_dataset, batch_size=batch_size, device=device, return_labels=True, return_thumbnails=True)
                stats = evaluator(features, labels)
                logging.info(evaluator.__class__.__name__ + ': ' + str(stats))
                writer.add_scalars(evaluator.__class__.__name__, stats, global_step=global_step)
                writer.add_embedding(features, metadata=labels, label_img=thumbnails, global_step=global_step)
                return stats

        else:
            def evaluate(global_step):
                features, labels = self.encode_dataset(val_dataset, batch_size=batch_size, device=device, return_labels=True)
                stats = evaluator(features, labels)
                logging.info(evaluator.__class__.__name__ + ': ' + str(stats))
                return stats
        if type(device) == str:
            device = torch.device(device)

        self.header.to(device)
        self.backbone.to(device)

        global_step = 0
        epoch = 0
        # [ys] add best eer condition
        best_eer = 1

        # Training per epoch
        while(True):
            start_time = time.time()
            logging.info(f"Epoch {epoch}:")
            train_losses = np.empty(len(train_dataloader))

            self.header.train()
            self.backbone.train()
            # [ys] add checkpoint
            if os.path.exists('checkpoint/checkpoint.pt'):
                loaded_checkpoint = torch.load('checkpoint/checkpoint.pt')
                checkpoint_epoch = loaded_checkpoint['epoch']
                trained_epoch = loaded_checkpoint['trained_epoch']
                print(f'checkpoint_{checkpoint_epoch} load complete!')
                self.backbone.load_state_dict(loaded_checkpoint['backbone_state_dict'])
                self.header.load_state_dict(loaded_checkpoint['header_state_dict'])
                optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
                best_eer = loaded_checkpoint.get('stats', {}).get('eer', best_eer)
            else:
                trained_epoch = 0

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for step, batch in pbar:
                # [ys] add break condition
                if any(item is None for item in batch):
                    print(f"Breaking batch due to None values: {batch}")
                    break
                # skip batch if singleton
                if len(batch[0]) <= 1:
                    continue

                # inputs = batch[0].to(device)
                inputs = batch[0].type(torch.FloatTensor).to(device)
                labels = batch[1].to(device).long().view(-1)

                features = self.backbone(inputs)
                # (batch, channel)
                features = features.view(features.shape[0], features.shape[1])

                outputs = self.header(features, labels)

                loss_value = self.loss(outputs, labels)
                train_losses[step] = loss_value

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                status = self.get_status_format_string(epoch, step, train_dataloader, global_step, max_epochs, max_training_steps, train_losses)
                pbar.set_description_str(status)

                if evaluator is not None and evaluation_steps > 0 and step % evaluation_steps == 0:
                    stats = evaluate(global_step)

                if tensorboard:
                    writer.add_scalar(self.loss.__class__.__name__, loss_value, global_step=global_step)

                global_step += 1
                if max_training_steps > 0 and global_step >= max_training_steps:
                    return


                if lr_global_step_scheduler is not None:
                    lr_global_step_scheduler.step()


            if evaluator is not None and max_epochs > 0 and evaluation_steps == 0:
                stats = evaluate(global_step)


            # [ys] add best eer condition
            if stats['eer'] < best_eer:
                print(f"epoch_eer : {stats['eer']}, best_eer : {best_eer}")
                best_eer = stats['eer']
                trained_epoch += 1
                # [ys] make checkpoint with torch
                checkpoint = {
                    'epoch': trained_epoch,
                    'backbone_state_dict': self.backbone.state_dict(),
                    'header_state_dict': self.header.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stats': stats,
                    'trained_epoch': trained_epoch
                }
                # [ys] save checkpoint
                checkpoint_path = 'checkpoint'
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                torch.save(checkpoint, f"{checkpoint_path}/checkpoint.pt")
                print(f'checkpoint_{trained_epoch} save complete!')
            else:
                trained_epoch += 1
                loaded_checkpoint['trained_epoch'] = trained_epoch
                torch.save(loaded_checkpoint, 'checkpoint/checkpoint.pt')
                print(f'train_epoch : {trained_epoch}')

            epoch += 1
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch_{epoch} Duration: {epoch_duration:.2f} seconds")
            # [ys] quit traning with condition
            if max_epochs > 0 and epoch >= max_epochs:
                return

            if lr_epoch_scheduler is not None:
                lr_epoch_scheduler.step()


    def get_status_format_string(self, epoch, step, train_dataloader, global_step, max_epochs, max_training_steps, train_losses):
        epoch_status = f"Epoch: {epoch+1}"
        status = [epoch_status]

        global_step_status = f"Global Step: {global_step}/{max_training_steps} ({global_step / max_training_steps * 100:.2f} %)"
        status.append(global_step_status)

        status.append(f"Train_Loss: {train_losses[max([0, step-self.loss_window+1]):step+1].mean()}")

        status = ' - '.join(status)
        return status


    def encode_dataset(self, dataset, batch_size, device, return_labels=False, return_thumbnails=False, thumbnail_size=32):
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        all_features = []
        all_labels = []
        all_thumbnails = []

        self.backbone.eval()
        with torch.no_grad():
            for step, batch in enumerate(dataloader):

                inputs = batch[0]
                labels = batch[1].view(-1)

                features = self.backbone(inputs.to(device))
                features = features.view(features.shape[0], features.shape[1]).cpu().numpy()

                all_features.extend([f for f in features])

                if return_labels:
                    all_labels.extend([l for l in labels.numpy()])

                if return_thumbnails:
                    all_thumbnails.extend([cv2.resize(img, dsize=(thumbnail_size, thumbnail_size), interpolation=cv2.INTER_CUBIC) for img in inputs.numpy().transpose(0, 2, 3, 1)])

        self.backbone.train()

        encoded = [np.array(all_features)]
        if return_labels:
            encoded.append(np.array(all_labels))
        if return_thumbnails:
            all_thumbnails = np.array(all_thumbnails)
            if len(all_thumbnails.shape) == 3:
                all_thumbnails = np.expand_dims(all_thumbnails, axis=-1)
            encoded.append(all_thumbnails.transpose(0, 3, 1, 2))

        return encoded
