import torch
import torch.nn as nn
from torchvision import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.utils import resample
from sklearn.utils import parallel_backend
import sklearn
sklearn.set_config(assume_finite=True)



class MLPModel(nn.Module):
    def __init__(self, dataset='cifar'):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        
        if dataset == 'cifar':
            input_dim = 3 * 32 * 32  
            num_classes = 10
        elif dataset == 'flower':
            input_dim = 3 * 224 * 224 
            num_classes = 5
        else:
            raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
            
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
        
      
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x



def create_model(model_name: str, dataset: str = 'cifar'):
    if model_name == 'mlp':
        return MLPModel(dataset)
    elif model_name == 'cnn':
        return CNNModel(dataset)
    elif model_name == 'cnn_svm':
        return CNNWithClassifier(dataset, classifier_type='svm')
    elif model_name == 'cnn_knn':
        return CNNWithClassifier(dataset, classifier_type='knn')
    elif model_name == 'cnn_dt':
        return CNNWithClassifier(dataset, classifier_type='dt')
    elif model_name == 'cnn_mlp':
        return CNNMLPModel(dataset)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



class CNNModel(nn.Module):
    def __init__(self, dataset='cifar'):
        super(CNNModel, self).__init__()
        
     
        if dataset == 'cifar':
            self.input_size = (32, 32)
            num_classes = 10
        elif dataset == 'flower':
            self.input_size = (224, 224)
            num_classes = 5
        else:
            raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
            
       
        self.features = nn.Sequential(
        
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
          
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
        
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
           
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
     
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
   
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
       
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x



class CNNWithClassifier(nn.Module):
    def __init__(self, dataset='cifar', classifier_type='svm'):
        super(CNNWithClassifier, self).__init__()
        
        if dataset == 'cifar':
            self.input_size = (32, 32)
            self.num_classes = 10
        elif dataset == 'flower':
            self.input_size = (224, 224)
            self.num_classes = 5
        else:
            raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
            
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        

        self.classifier_type = classifier_type
        if classifier_type == 'knn':
            if dataset == 'cifar':
                self.classifier = KNeighborsClassifier(
                    n_neighbors=8,  
                    weights='distance',
                    metric='minkowski',
                    p=2,  
                    n_jobs=-1,
                    algorithm='ball_tree'  
                )
            else:
                self.classifier = KNeighborsClassifier(
                    n_neighbors=5,
                    weights='uniform',
                    n_jobs=-1
                )
        elif classifier_type == 'svm':
            if dataset == 'cifar':
                self.classifier = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='auto',
                    probability=True,   
                    cache_size=2000,
                    class_weight='balanced',
                    decision_function_shape='ovr',
                    random_state=42
                )
            else:
                self.classifier = SVC(
                    kernel='rbf',
                    probability=True,
                    cache_size=2000
                )
        elif classifier_type == 'dt':
            if dataset == 'cifar':
                self.classifier = DecisionTreeClassifier(
                    criterion='gini',
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42
                )
            else:
                self.classifier = DecisionTreeClassifier()
        
        self.is_sklearn_classifier = True
        self.trained = False
        self.feature_buffer = []
        self.label_buffer = []
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.dim_reduction(x) 
        x = self.avgpool(x)
        x = self.flatten(x)
        
        if not self.trained:
            return x
            
        features_np = x.detach().cpu().numpy()
        
        if hasattr(self.classifier, 'predict_proba'):
            predictions = self.classifier.predict_proba(features_np)
        else:
            predictions = self.classifier.predict(features_np)
            
        return torch.from_numpy(predictions).float().to(x.device)
    
    def update_features(self, features, labels):
        self.feature_buffer.append(features.detach().cpu().numpy())
        self.label_buffer.append(labels.detach().cpu().numpy())
    
    def fit_classifier(self):
        
        if len(self.feature_buffer) > 0:
            X = np.vstack(self.feature_buffer)
            y = np.concatenate(self.label_buffer)
            
            with parallel_backend('threading', n_jobs=-1):
                self.classifier.fit(X, y)
            
            self.trained = True
            self.feature_buffer = []
            self.label_buffer = []

class CNNMLPModel(nn.Module):
    def __init__(self, dataset='cifar'):
        super(CNNMLPModel, self).__init__()
        
        
        if dataset == 'cifar':
            self.input_size = (32, 32)
            num_classes = 10
        elif dataset == 'flower':
            self.input_size = (224, 224)
            num_classes = 5
        else:
            raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
        
        self.cnn_features = nn.Sequential(
          
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
           
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
       
        self.mlp = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.cnn_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.mlp(x)
        return x