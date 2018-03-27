# Reference

```
@ARTICLE{koon,
	author  = {Xu, H and Zhao, C G}, 
	title   = {Softer loss transfer distilling and saliency detection for image-based plant diseases diagnosis}, 
	journal = {}, 
	year    = {2018},
	volume  = {}, 
	pages   = {}
}
```

# Licence
All the codes and data are licensed under Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) license
<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>.
# Original Dataset
https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge

# Environment preparation
Tested environment: Ubuntu 16.04 + CIDA 9.1 + CuDNN 7.05
Tested python version: python 3.5
Recommend method: [Virtualenv+CUDA+CuDNN](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/install/install_linux.md)
CUDA 9.1 contains driver 390. It's better to install it via network version, because network version contains the newies compatible driver.
If you have any trouble when installing CUDA, the best way to look for help is to read the NVIDA installation document first.

## packages
[Pytorch 0.3.1](https://github.com/pytorch/pytorch#installation) + packages in the requirements.txt

```shell
pip install -r requirements.txt
```
# Deployement Example
Given an image, the diagnosis returns the probability of the predicted class and the classification saliency map.

`05_client_example.ipynb`

# Experiment Procedure
## 0 Dataset preprocessing
Download the dataset from the above archived doi links.
(save your dataset to SSD will saves alot of time)
### 0.1 replicate our experiments
We have already prepared the data separation in `.txt` files. You only need to do
```
$ python change_filename_prefix.py --prefix /home/usrname/plantvillage_deeplearning_paper_dataset
```
where `change_filename_prefix.py` is in the archived dataset. 

### 0.2 Do your own experiments 
You can also generate new training and testing dataset as thoses in `80-20` like directories in the dataset archive.
1. extracting the archived dataset.
2. `cd` to the extracted dataset direcotry (e.g. `/home/usrname/plantvillage_deeplearning_paper_dataset`) that contains `color` and `filtered_leafmaps` directories.
3. run `jupyter notebook`
4. open `gen_data.ipynb` and change the
   * `INPUT_FOLDER` to `/home/usrname/plantvillage_deeplearning_paper_dataset/color`
   * `OUTPUT_FOLDER` to `/home/usrname/plantvillage_deeplearning_paper_dataset`
   * `TEST_PERCENT` to what you require
   * `data_separate` to `str(1-TEST_PERCENT)-TEST_PERCENT`
5. run `gen_data.ipynb` line by line.

## 1 Initial last layer transfer learning to decide the transfer learning model structure
* a. run `python initial_transferlearn_gen_bottleneck.py -h` to see the parameters we can set.
* b. run `python initial_transferlearn_gen_bottleneck.py --dataroot Your_Local_Data_Root_Dir` to set the the new data directory, and start generating the bottleneck output of the entire dataset over ResNet18, ResNet34, ResNet50, ResNet101 and ResNet152 models.
* c. run `python initial_transferlearn_train.py --dataroot Your_Local_Data_Root_Dir --saveroot Your_Local_Model_Save_root --dataseparate '50-50'` to training the last layer of 5 resnet model using the Plant Village dataset.
Here we are using 50-50 dataset split. In the papaer we also test 20-80 separation.
* d. open `01bottleneck_transfer.ipynb`, to see the testing results.

## 2 (Optional) Generating the (1024, 14, 14) ResNet101 middle layers' output from input images
This takes about 50 GB spaces.
* open `02leaf_resnet101_transfer_preprosessing.ipynb` to see the results.

## 3 Further transfer learning
### 3.1 (Optional) Partial ResNet101
Change the `--dataroot` to the path where you saved the downloaded dataset, and `--saveroot` to the path that you want to save the models(can be a mobile hard disk).
#### 3.1.1 Original cross-entropy
```shell
nohup ./train_resnet101_org_loss.sh &
```
#### 3.1.2 weighted cross-entropy
```shell
nohup ./train_resnet101_weighted_loss.sh &
```
#### 3.1.3 softer cross-entropy
```shell
nohup ./train_resnet101_softer_loss.sh &
```

### 3.2 ResNet18
Change the `--dataroot` to the path where you saved the downloaded dataset, and `--saveroot` to the path that you want to save the models(can be a mobile hard disk).
#### 3.2.1 Original cross-entropy
```shell
nohup ./train_resnet18_org_loss.sh &
```
#### 3.2.2 weighted cross-entropy
```shell
nohup ./train_resnet18_weighted_loss.sh &
```
#### 3.2.3 softer cross-entropy
```shell
nohup ./train_resnet18_softer_loss.sh &
```

## 4 Model averaging
Saving all classification scores of the testing datasets. This is used as the evaluation of the model averaging power.
```shell
nohup ./save_resnet18_testing_logits.sh &
nohup ./save_resnet101_testing_logits.sh &
``` 

## 5 Distilling
* Saving the all classification scores of the five experiments training datasets. Which will be used as the softer probability loss targets in distilling.

### 5.1 Original distilling
* Distilling ResNet18 models using the averaged classification scores.
```shell
nohup ./train_resnet18_orig_distill.sh &
nohup ./train_resnet18_orig_distill.sh &
``` 
### 5.2 Modified distilling
```shell
nohup ./train_resnet18_modified_distill.sh &
``` 

### 5.3 Save logits
```shell
./save_resnet18_distill_testing_logits.sh
./save_resnet18_distill_training_logits.sh
```

Second run to see the results.
```shell
./save_resnet18_distill_testing_logits.sh
./save_resnet18_distill_training_logits.sh
```
## 6 Results and visualization
* open `04leaf_assess.ipynb`


