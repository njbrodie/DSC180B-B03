# DSC180B-B03

How to run the code:

For the GP model it is necessary to run the GP_vpd_preprocess.ipynb before running the GP_vpd.ipynb notebook. The preprocessing file will create the necessary data files that will be used to train and test the GP model.

The data needed for to run the linear and random forest models can be found in the data folder of this Github, or by using the vpd_preprocess.ipynb notebook if you have access to NCAR. The data must be placed into a folder in your working directory titled vpd_data.

The CNN model can be trained by using the CNN_run.py file which takes in your desired input simulations(ssp126, ssp370, etc.) separated by a single space. For example, the command python3 CNN_run.py scenario_1 scenario_2 scenario_3 will train the CNN with the appropriate data files and use CNN_helper.py to train the model on the desired scenarios and return the rmse values.

Here is a link to our website: https://njbrodie.github.io/DSC180B-B03/


