# ERNIE text classification by PyTorch

This repo contains a PyTorch implementation of a pretrained ERNIE model  for text classification.

## Structure of the code

At the root of the project, you will see:

```text
├── pyernie
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  |  └── pretrain　
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── utils # a set of utility functions
├── convert_ernie_to_pytorch.py
├── fine_tune_ernie.py
```
## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib
- tensorboardX
- Tensorflow (to be able to run TensorboardX)

## How to use the code

you need download pretrained ERNIE model

1. Download the  pretrained ERNIE model from [baiduNetDisk](https://pan.baidu.com/s/1BQlwbc9PZjAoVB7Kfq_Ihg) {code: uwds} and place it into the `/pyernie/model/pretrain` directory.

2. prepare Chinese raw data(example,news data), you can modify the `io.data_transformer.py` to adapt your data.

3. Modify configuration information in `pyernie/config/basic_config.py`(the path of data,...).

4. run `fine_tune_ernie.py`.

## Fine-tuning result

### training 

Epoch: 4 -  loss: 0.0136 - f1: 0.9967 - valid_loss: 0.0761 - valid_f1: 0.9798

### train classify_report

|    label    | precision | recall | f1-score | support |
| :---------: | :-------: | :----: | :------: | :-----: |
|     财经      |   0.99    |  0.99  |   0.99   |  3500   |
|     体育      |   1.00    |  1.00  |   1.00   |  3500   |
|     娱乐      |   1.00    |  1.00  |   1.00   |  3500   |
|     家居      |   1.00    |  1.00  |   1.00   |  3500   |
|     房产      |   0.99    |  0.99  |   0.99   |  3500   |
|     教育      |   1.00    |  0.99  |   1.00   |  3500   |
|     时尚      |   1.00    |  1.00  |   1.00   |  3500   |
|     时政      |   1.00    |  1.00  |   1.00   |  3500   |
|     游戏      |   1.00    |  1.00  |   1.00   |  3500   |
|     科技      |   0.99    |  1.00  |   1.00   |  3500   |
| avg / total   |   1.00   |  1.00  |   1.00   |  35000  |

### valid classify_report

|    label    | precision | recall | f1-score | support |
| :---------: | :-------: | :----: | :------: | :-----: |
|     财经      |   0.97    |  0.96  |   0.96   |  1500   |
|     体育      |   1.00    |  1.00  |   1.00   |  1500   |
|     娱乐      |   0.99    |  0.99  |   0.99   |  1500   |
|     家居      |   0.99    |  0.99  |   0.99   |  1500   |
|     房产      |   0.96    |  0.96  |   0.96   |  1500   |
|     教育      |   0.98    |  0.98  |   0.98   |  1500   |
|     时尚      |   0.99    |  0.99  |   0.99   |  1500   |
|     时政      |   0.97    |  0.98  |   0.98   |  1500   |
|     游戏      |   0.99    |  0.99  |   0.99   |  1500   |
|     科技      |   0.97    |  0.97  |   0.97   |  1500   |
| avg / total |   0.98    |  0.98  |   0.98   |  15000  |

### training figure

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190519002915.png)

## Tips

- When converting the tensorflow checkpoint into the pytorch, it's expected to choice the "bert_model.ckpt", instead of "bert_model.ckpt.index", as the input file. Otherwise, you will see that the model can learn nothing and give almost same random outputs for any inputs. This means, in fact, you have not loaded the true ckpt for your model
- When using multiple GPUs, the non-tensor calculations, such as accuracy and f1_score, are not supported by DataParallel instance
- As recommanded by Jocob in his paper <url>https://arxiv.org/pdf/1810.04805.pdf<url/>, in fine-tuning tasks, the hyperparameters are expected to set as following: **Batch_size**: 16 or 32, **learning_rate**: 5e-5 or 2e-5 or 3e-5, **num_train_epoch**: 3 or 4
- The pretrained model has a limit for the sentence of input that its length should is not larger than 512, the max position embedding dim. The data flows into the model as: Raw_data -> WordPieces -> Model. Note that the length of wordPieces is generally larger than that of raw_data, so a safe max length of raw_data is at ~128 - 256 
- Upon testing, we found that fine-tuning all layers could get much better results than those of only fine-tuning the last classfier layer. The latter is actually a feature-based way 
