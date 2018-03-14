
# Europe Parliament Dataset - language detection task

 - These are my baseline results for language detection.

 - Features Used 
	 - Basic ASCII representations have been used to represent the sentences. 

Further parameter optimization can lead to better results. But I am settling with these results for now. I plan to add more features and then evaluate the model.


## Usage
**train_xgb.py** - Change *my_dir* variable to the folder in which the data is present.

Running this file will save the model in ***basic.model*** file.

**test_predictions.py** - Change *test_data_path* to test data file name

Running this file will evaluate the model trained on the test data and generate a classification report.

Links to Datasets used -
Train Data - http://www.statmt.org/europarl/v7/europarl.tgz
Test Data - https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/language-detection/europarl-test.zip
## Results

Validation Set Classification Report (using XGBoost)

	 precision    recall  f1-score   support

         bg       0.98      1.00      0.99       688
         cs       0.93      0.93      0.93       867
         da       0.95      0.91      0.93       920
         de       0.97      0.94      0.95       908
         el       0.99      1.00      1.00       885
         en       0.81      0.98      0.89       976
         es       0.95      0.91      0.93       982
         et       0.96      0.94      0.95       822
         fi       0.97      0.96      0.97       952
         fr       0.96      0.95      0.96       990
         hu       0.97      0.96      0.97       869
         it       0.92      0.95      0.94       958
         lt       0.95      0.96      0.96       869
         lv       0.98      0.98      0.98       886
         nl       0.93      0.93      0.93       946
         pl       0.99      0.98      0.98       859
         pt       0.97      0.92      0.94       980
         ro       0.97      0.93      0.95       658
         sk       0.92      0.88      0.90       888
         sl       0.95      0.94      0.95       857
         sv       0.94      0.95      0.94       948

	avg / total       0.95      0.95      0.95     18708


Test Data Report

	   precision    recall  f1-score   support

         bg       1.00      1.00      1.00      1000
         cs       0.61      0.62      0.61      1000
         da       0.69      0.46      0.55      1000
         de       0.81      0.47      0.59      1000
         el       1.00      1.00      1.00      1000
         en       0.41      0.95      0.57      1000
         es       0.61      0.35      0.45      1000
         et       0.56      0.55      0.56      1000
         fi       0.81      0.65      0.72      1000
         fr       0.67      0.72      0.69      1000
         hu       0.75      0.69      0.72      1000
         it       0.58      0.61      0.60      1000
         lt       0.71      0.67      0.69      1000
         lv       0.73      0.95      0.83      1000
         nl       0.53      0.70      0.60      1000
         pl       0.86      0.91      0.89      1000
         pt       0.69      0.29      0.41      1000
         ro       0.74      0.70      0.72      1000
         sk       0.54      0.42      0.47      1000
         sl       0.72      0.63      0.67      1000
         sv       0.57      0.79      0.66      1000

	avg / total       0.69      0.67      0.67     21000
