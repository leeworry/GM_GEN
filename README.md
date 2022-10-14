# GM_GEN
The code of the paper *"Graph-based Model Generation for Few-Shot Relation Extraction"*. This paper has been accepted to EMNLP2022.
You can find the main results (**username is GM_GEN**) in the paper on FewRel 1.0 competition on CodaLab competition websit: [FewRel 1.0 Competition](https://competitions.codalab.org/competitions/27980#results)

We will release the paper link after camera ready.

### Environments
- ``python 3.7.10``
- ``PyTorch 1.7.1``
- ``transformers 4.7.0``

### Datasets and Models
You can find the training and validation data here: [FewRel 1.0 data](https://github.com/thunlp/FewRel/tree/master/data). For the test data, you can easily download from FewRel 1.0 competition website: https://competitions.codalab.org/competitions/27980


### Code
Put all data in the **data** folder, CP pretrained model in the **CP_model** folder (you can download CP model from https://github.com/thunlp/RE-Context-or-Names/tree/master/pretrain or [Google Drive](https://drive.google.com/drive/folders/1AwQLqlHJHPuB1aKJ8XPHu8nu237kgtWj?usp=sharing))

#### Train
You can train a 5-way-1-shot model by:
```
python train_demo.py --trainN 5 --N 5 --K 1 --Q 1 --pretrain_ckpt ../Bert4SemiRE/bert/BERT-BASE-UNCASED --cat_entity_rep --model gm_gen --dot
```
or a 5-way-5-shot model by:
```
python train_demo.py --trainN 5 --N 5 --K 5 --Q 1 --pretrain_ckpt ../Bert4SemiRE/bert/BERT-BASE-UNCASED --cat_entity_rep --model gm_gen
```

#### Evaluation
The learning rate of the generated model will affect the model performance, so we recommend that you select an appropriate learning rate through the valiadation set before setting the learning rate by:

```
python train_demo.py --trainN 5 --N 5 --K 5 --Q 1 --pretrain_ckpt ../Bert4SemiRE/bert/BERT-BASE-UNCASED --cat_entity_rep --model gm_gen --load_ckpt ./checkpoint/gm_gen-bert-train_wiki-val_wiki-5-1-dot-catentity.pth.tar --param_lr 0.0001 --test_params --only_test
```
Then, an online test can be performed by:
```
python train_demo.py --trainN 5 --N 5 --K 5 --Q 1 --pretrain_ckpt ../Bert4SemiRE/bert/BERT-BASE-UNCASED --cat_entity_rep --model gm_gen --test_online --load_ckpt ./checkpoint/gm_gen-bert-train_wiki-val_wiki-5-1-dot-catentity.pth.tar --test_lr 0.0001 --only_test
```

Some explanations of the parameters in the script:
```
--load_ckpt
	the path of the trained model
--pretrain_ckpt
	the path of the pretrained model
--param_lr
	the learning rate of the generated models on validation set
--test_lr
	the learning rate of the generated models on test set
```

### Results

**BERT on FewRel 1.0**

|                   | 5-way-1-shot | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 92.65 | 95.62 | 86.81 | 91.27 |
| Test              | 94.89 | 96.96 | 91.23 | 94.30 |

**CP on FewRel 1.0**

|                   | 5-way-1-shot | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 96.97 | 98.32 | 93.97 | 96.58 |
| Test              | 97.03 | 98.34 | 94.99 | 96.91 |
