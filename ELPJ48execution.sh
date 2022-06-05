cd ./CHD_49/ELPJ48/ 
tar xvzf scikit_ml_learn_data.zip
python3.5 execution_script.py CHD_49 ELPJ48 test
python3.5 parseJsonFiles.py CHD_49 ELPJ48 
rm -r ProcessedDatasets 
rm -r scikit_ml_learn_data 
rm scikit_ml_learn_data.tar.gz 
tar -czf results_CHD_49_ELPJ48.zip *
