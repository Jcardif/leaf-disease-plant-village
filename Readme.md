## Licence
All the codes and data are licensed under Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) license
<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>.
## Original Dataset
https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge

# Experiment Procedure
## 0 Dataset preprocessing
**Do need to run again.**
The `.txt` file for training and testing are already generated.
We are also generated a validation set. But when generating the dataset,
we combine the `validate.txt` and `test.txt` file as the testing dataset.

## 1 Initial last layer transfer learning to decide the transfer learning model structure
* a. run `python initial_transferlearn_gen_bottleneck.py -h` to see the parameters we can set.
* b. run `python initial_transferlearn_gen_bottleneck.py --dataroot Your_Local_Data_Root_Dir` to set the the new data directory, and start generating the bottleneck output of the entire dataset over 5 resnet model.
* c. run `python initial_transferlearn_train.py --saveroot Your_Local_Model_Save_root --dataseparate '50-50'` to training the last layer of 5 resnet model using the Plant Village dataset.
Here we are using 50-50 dataset split.
* d. open `01bottleneck_transfer.ipynb`, to see the testing results.

## 2 Generating the (1024, 14, 14) ResNet101 middle layers' output from (3,64,64) input images
* open `02leaf_resnet101_transfer_preprosessing.ipynb` to see the results.

## 3 Further transfer learning using softer probability loss




## 4 Model averaging
* run `save_resnet18_testing_logits.sh` to save the all classification scores of the five experiments testing datasets. This is used as the evaluation of the model averaging power.
* open `04leaf_assess.ipynb` to see the reuslts of softer probability loss.
* open `04leaf_assess.ipynb` to see the reuslts of the model averaging using arithmatic mean of the testing dataset classification scores from the former steps.

## 5 Distilling
* run `save_resnet18_training_logits.sh` to save the all classification scores of the five experiments training datasets. Which will be used as the softer probability loss targets in distilling.
* run `train_distilled_model.sh` to distilling ResNet18 models using the averaged classification scores.

## 6 Visualization
* open `04leaf_assess.ipynb` to see the reuslts of the model visualization.

## 7 A toy client in deployment


