# Data Generation Scripts

This set of scripts are used to start the MLC methods. 

1. algorithmConfiguration.py has the methods implementations invocation
2. dataSetsPreprocessing.py implements utilities for reading and processing the data
3. evaluation_script.py implements the performance criteria
4. execution_script.py implements an experiment that involves   
5. parseJsonFiles.py parses the outputs in .tar.gz file for storage



## 1. algorithmConfiguration.py
Output: Creates file containing all aglortihm configurations.


## 2. dataSetsPreprocessing.py
Input: MLL datasets.
This sorts all the files according to the metric: #features * #targets * #numberSamples.
It loads dataset by dataset. 
Supervisor adds the properties of each of the datasets.
The datasets should be in BigEndian format. The BigEndian refers that frist the target are present then the features in the .arff files.
Output: Sorted file containg the datasets in ascending order given the metric: #features * #targets * #numberSamples

Exmaple execution file for the method ELPJ48 on the dataset CHD_49 is given in ELPJ48execution.sh. 
The help file "scikit_ml_learn_data.tar.gz" can be downloaded from: https://tubcloud.tu-berlin.de/s/2qSsDkcCoBRDYxP
