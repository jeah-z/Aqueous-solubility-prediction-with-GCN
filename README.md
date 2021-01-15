# Demonstration for the training & test errors in the publication :
- @article{gao2020accurate,
  title={Accurate predictions of aqueous solubility of drug molecules via the multilevel graph convolutional network (MGCN) and SchNet architectures},
  author={Gao, Peng and Zhang, Jie and Sun, Yuzhu and Yu, Jianguo},
  journal={Physical Chemistry Chemical Physics},
  volume={22},
  number={41},
  pages={23766--23772},
  year={2020},
  publisher={Royal Sossciety of Chemistry}
}
 
For the training & test errors presented in the publication, we need to claim that, for Delaney and Huuskonen data sets, the validation errors were 0.27 and 0.23, respectively; and these results can be reproduced via the provided code and original data sets. It is also alternative for users to incorporate the validation data in their training, thus the corresponding errors can be further reduced to 0.05 and 0.06, respectively. 
ssss

# Aqueous-solubility-prediction-with-GCN

A code to predict aqueous solubility.

This code was based on https://github.com/tencent-alchemy/Alchemy. If this script is of any help to you, please cite them.

K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) link
- @article{chen2019alchemy,
  title={Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models},
  author={Chen, Guangyong and Chen, Pengfei and Hsieh, Chang-Yu and Lee, Chee-Kong and Liao, Benben and Liao, Renjie and Liu, Weiwen and Qiu, Jiezhong and Sun, Qiming and Tang, Jie and Zemel, Richard and Zhang, Shengyu},
  journal={arXiv preprint arXiv:1906.09427},
  year={2019}
}

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
python train.py --model sch --epochs 20000 --dataset Dataset/delaney  --save YOUR_DIR_TO_SAVE_RESULTS
```

## Eval and cross validation

The scripts to evaluate trained model have been included in the directory Scripts, which is left to be explore by users.