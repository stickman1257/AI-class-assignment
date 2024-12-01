import os
import argparse
import wandb
import numpy as np
import cv2 as cv2
import pandas as pd
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from src.model import CNNWithClassifier
from src.dataset import CustomDataModule
from src.model import create_model


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


from sklearn.utils import parallel_backend
import sklearn
sklearn.set_config(assume_finite=True)

SEED = 36
L.seed_everything(SEED)

class ClassficationModel(L.LightningModule):
    def __init__(self,model, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses = []
        self.labels = []
        self.predictions = []

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        if isinstance(self.model, CNNWithClassifier) and self.model.is_sklearn_classifier:
            features = self.model(inputs)
            self.model.update_features(features, labels)
            return {'loss': torch.tensor(0.0, requires_grad=True)}
        else:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            self.log('train_loss', loss)
            return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        
        if isinstance(self.model, CNNWithClassifier) and self.model.is_sklearn_classifier:
            if not self.model.trained:
                self.model.update_features(outputs, labels)
                return 0.0
            
            if len(outputs.shape) > 1:
                _, predictions = torch.max(outputs, 1)
            else:
                predictions = outputs
                
            accuracy = (predictions.cpu().numpy() == labels.cpu().numpy()).mean()
        else:
            loss = self.loss_fn(outputs, labels)
            _, predictions = torch.max(outputs.data, 1)
            accuracy = (predictions == labels).float().mean()
            self.losses.append(loss.item())
            
        self.labels.append(labels.cpu().numpy())
        self.predictions.append(predictions.cpu().numpy())
        
        self.log('val_acc', accuracy)
        return accuracy
    
    def on_validation_epoch_end(self):
        if isinstance(self.model, CNNWithClassifier) and self.model.is_sklearn_classifier:
            if not self.model.trained:
                self.model.fit_classifier()
                return
                
        labels = np.concatenate(self.labels)
        predictions = np.concatenate(self.predictions)
        acc = (labels == predictions).mean()
        
        if not (isinstance(self.model, CNNWithClassifier) and self.model.is_sklearn_classifier):
            loss = sum(self.losses)/len(self.losses) if self.losses else 0
            self.log('val_epoch_loss', loss)
            
        self.log('val_epoch_acc', acc)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        
        if isinstance(self.model, CNNWithClassifier) and self.model.is_sklearn_classifier:
            target_np = target.cpu().numpy()
            if torch.is_tensor(output):
                if len(output.shape) > 1:
                    _, predictions = torch.max(output, 1)
                    predict_np = predictions.cpu().numpy()
                    loss = self.loss_fn(output, target)
                else:
                    predict_np = output.cpu().numpy()
                    
                    output_tensor = torch.zeros(len(target), self.model.num_classes, device=target.device)
                    output_tensor[range(len(target)), predict_np] = 1
                    loss = self.loss_fn(output_tensor, target)
            else:
                predict_np = output
                output_tensor = torch.zeros(len(target), self.model.num_classes, device=target.device)
                output_tensor[range(len(target)), predict_np] = 1
                loss = self.loss_fn(output_tensor, target)
            
            self.labels.append(target_np)
            self.predictions.append(predict_np)
        else:
            loss = self.loss_fn(output, target)
            _, predictions = torch.max(output, 1)
            
            target_np = target.detach().cpu().numpy()
            predict_np = predictions.detach().cpu().numpy()
            
            self.labels.append(target_np)
            self.predictions.append(predict_np)
        
        self.losses.append(loss)
        self.log('test_loss', loss)
        return loss
    
    def on_test_epoch_end(self):
        labels = np.concatenate(self.labels)
        predictions = np.concatenate(self.predictions)
        
     
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        
   
        acc = np.mean(labels.astype(np.int32) == predictions.astype(np.int32))
        
   
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        wandb.log({"test_confusion_matrix": wandb.Image(plt)})
        plt.close()
        
        print("\nTest Classification Report:")
        print(classification_report(labels, predictions))

    
        loss = torch.stack(self.losses).mean() if self.losses else torch.tensor(0.0)

        self.log('test_epoch_acc', acc)
        self.log('test_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()

    def predict_step(self, batch, batch_idx):
        inputs, img = batch
        output = self.model(inputs)
        _, pred_cls = torch.max(output, 1)

        return pred_cls.detach().cpu().numpy(), img

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)    

def main(classification_model, data, batch, epoch, save_path, device, gpus, precision, mode, ckpt, dataset):
    model = ClassficationModel(
        create_model(
            model_name=classification_model, 
            dataset=dataset 
        ),
        batch_size=batch
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if device == 'gpu':
        if len(gpus) == 1:
            gpus = [int(gpus)]
        else:
            gpus = list(map(int, gpus.split(',')))
    elif device == 'cpu':
        gpus = 'auto'
        precision = 32
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_acc',
        mode='max',
        dirpath= f'{save_path}',
        filename= f'{classification_model}-'+'{epoch:02d}-{val_epoch_acc:.2f}',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_epoch_acc',
        mode='max',
        patience=10
    )
    wandb_logger = WandbLogger(project="인공지능 과제 11.29_1")
    
    if mode == 'train':
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            max_epochs=epoch,
            precision=precision,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
        )
        data_module = CustomDataModule(
            dataset_type='cifar10' if dataset == 'cifar' else 'flowers',
            data_path=data,
            batch_size=batch
        )
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    else:
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            precision=precision
        )
        model = ClassficationModel.load_from_checkpoint(
            ckpt, 
            model=create_model(classification_model, dataset)
        )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='resnet')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='./data/')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-p', '--precision', type=str, default='32-true')
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar', choices=['cifar', 'flower'], 
                       help='Choose dataset: cifar or flower')
    args = parser.parse_args()
    
    main(args.model, args.data, args.batch, args.epoch, args.save, args.device, args.gpus, args.precision, args.mode, args.ckpt, args.dataset)