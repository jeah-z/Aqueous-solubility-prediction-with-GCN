# Aqueous-solubility-prediction-with-GCN



## Dataset preprocessing
Dataset can be processed with the following command:
```
python dataset_split.py --dataset Dataset/delaney  
```
The dataset will be normalized and splitted, 90% of which as training dataset and 10% as validation dataset.
![image](https://github.com/jeah-z/Drug-Solubility-Prediction-Mordred/blob/master/Figures/dnn.png)
Fig. 1 Architecture of solubility prediction models based on DNN
![image](https://github.com/jeah-z/Drug-Solubility-Prediction-Mordred/blob/master/Figures/gcn.png)
Fig. 2 Architecture of solubility prediction models based on GCN



## Training
First, the dictionary with the name of dataset  should be created to keep the saved model:
```
mkdir delaney
```
The model can be trained with the following command:
```
python train.py --model sch --epochs 20000 --dataset delaney  
```

