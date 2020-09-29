# Aqueous-solubility-prediction-with-GCN



## Dataset preprocessing
Dataset can be processed with the following command:
```
python dataset_split.py --dataset Dataset/delaney  
```
The dataset will be normalized and splitted, 90% of which as training dataset and 10% as validation dataset. It is worthed noticed that there are ~400 duplicated compouds in huus and delaney datasets.



## Training
First, the dictionary with the name of dataset  should be created to keep the saved model:
```
mkdir delaney
```
The model can be trained with the following command:
```
python train.py --model sch --epochs 20000 --dataset delaney  
```

## Eval and cross validation

The scripts to evaluate trained model have been included in the directory Scripts, which is left to be explore by users.