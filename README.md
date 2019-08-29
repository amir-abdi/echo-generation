## GAN-enhanced Conditional Echocardiogram Generation   

This repository accompanies our submission to the 
Med-NeurIPS 2019 workshop of 
the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.


#### Training

This is a Python3 implementation. To train the models, first install the requirements by running

    pip3 install -r requirements.txt

Subsequently, to train the model call `main.py` with a 
config file of your choosing. 
The 5 config files corresponding to the 5 experiments of the article are
available in the `configs/` directory.   

     python3 src/main.py \
     --dataset_path=$DATASETS/CAMUS \
     --config=configs/ventricle.json
     
The environment variable `$DATASET` is assumed to be set to 
where the CAMUS dataset directory is stored. 

#### Sample Generated Echos

![Reconstructed Samples](./imgs/results.png)