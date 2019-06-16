# Safety Challenge  


### Author
by: Xavier M. Puspus  
Email: xpuspus@gmail.com  
Country: Philippines  

### Based on telematics data, how might we detect if the driver is driving dangerously?  
Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, derive a model that can detect dangerous driving trips.

### Details

The given dataset contains telematics data during trips (bookingID). Each trip will be assigned with label 1 or 0 in a separate label file to indicate dangerous driving. Pls take note that dangerous drivings are labelled per trip, while each trip could contain thousands of telematics data points. participants are supposed to create the features based on the telematics data before training models.

# About the Repository

### Notebook
The notebook `safety_challenge_xmpuspus.ipynb` is the main notebook for the challenge. It contains codes for loading the data, feature engineering, model training, model scoring and hold-out scoring. To run the notebook, please place the dataset in the `data/safety/` folder in their corresponding directories.

### Folders
The folder `data/` contains all data used for this challenge. It contains the `safety` folder as downloaded from the grab AIforSEA Safety Challenge. The folder `model/` contains the pickled trained model should the examiner choose to just measure on holdout set. Please see last section of this ReadMe for details on how to model on holdout set. 



### Model Approach
I trained a classifier on the first 120 seconds of each feature/signal, concatenated into a single array and fed it into a feed-forward multi-layer neural network. Details of the data pre-process is in the notebook stated above as well as in the `utils/utils.py` script in the function `process_data()`.


# Model Prediction on Holdout Set

**To examiner**, please save hold out data to `data/test/` folder with the same folder structure as the one provided for the challenge in `safety/` folder (hold out features should be in a folder named `features/` and the labels should be in a folder `labels/`, where both folders should be placed in `data/test/` folder). Run cells below the header **Measure From Holdout Data** once the holdout data is in the suggested folder.
