#!/bin/bash


echo "Starting CIFAR-10 experiments..."
python main.py -m mlp -mo train -ds cifar
python main.py -m cnn -mo train -ds cifar
python main.py -m cnn_knn -mo train -ds cifar
python main.py -m cnn_svm -mo train -ds cifar
python main.py -m cnn_dt -mo train -ds cifar
python main.py -m cnn_mlp -mo train -ds cifar


echo "Starting Flower experiments..."
python main.py -m mlp -mo train -ds flower
python main.py -m cnn -mo train -ds flower
python main.py -m cnn_knn -mo train -ds flower
python main.py -m cnn_svm -mo train -ds flower
python main.py -m cnn_dt -mo train -ds flower
python main.py -m cnn_mlp -mo train -ds flower

echo "All experiments completed!"