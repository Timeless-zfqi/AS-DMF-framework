# AS-DMF：A Lightweight TLS encrypted traffic detection framework
Authors:   
## Contents
- [Introduction](#Introduction)
- [Setup](#Setup)
- [Dataset and feature extraction](#Dataset-and-feature-extraction)
- [Feature reduction mechanism](#Feature-reduction-mechanism)
- [DMF classifier](#DMF-classifier)
- [Query and training](#Query-and-training) 
- [Acknowledgement](#Acknowledgement) 

## Introduction  
Our project is a combination of active learning and feature reduction to achieve lightweight detection of TLS encrypted malicious traffic. The aim is to work lightly on both data and feature dimensions.  
__Modules of AS-DMF framework include:__
* __Data pre-processing and feature extraction__.
This module is used to pre-process the captured pcap packets and perform preliminary feature extraction to select the TLS encrypted flows to form the initial sample set.
* __Feature reduction mechanism__.
This module is used to perform feature selection and to study feature-level lightweighting.
* __DMF classifier__.
DMF classifier is the model used to train query samples in AS-DMF framework.
* __Query and train__.
This module is the query and training process of AS-DMF. It mainly uses the pool-based active learning framework and specific querying strategies to query and label informative and representative instances. and train the labeled samples using DMF classifier.
![DFM](https://github.com/Timeless-zfqi/AS-DMF-framework/blob/main/Figure/Framework.jpg)

## Setup
Before you use this project, you must configure the following environment.  
1. Requirements
```
python >= 3.7
linux >= Ubuntu 20.04
zeek(LST) >= 4.0+
wireshark
```  
2. Basic Dependencies
```
scikit-learn
zat
zeek-flowmeter
alipy
```  
3. Others  
For other packets used in the experiment, please refer to _impot.txt_
## Dataset and feature extraction
You can run this module in _Data pre-processing.ipynb_. Details are shown below:   

1.Dataset  
We use the open source [CTU-13](https://www.stratosphereips.org/datasets-ctu13 "CTU-13") botnet dataset.

2.How to merge pacp packets?  
You need to execute the following command from the command line:
```
>cd wireshark
>mergecap -w target_path/normal.pcap source_path/CTU-Normal/*.pcap
```
3. Initial feature extraction in zeek  
```
zeek flowmeter -C -r target pcap path/*.pcap (or .pcapng is also accept)
```
4. To Python  
Import the extracted features into Python by zat and filter the TLS encrypted flows.  

## Feature reduction mechanism
Use ANOVA and MIC to sort the features and pick the number of features you need.  
You can run this module in the _feature reduction mechanism.ipynb_.  
```python
n = number
X_reduction = SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:mic(x, Y), 
                    X.T))).T)),k=n).fit_transform(x_de,y)
```
## DMF classifier  
### Structure  
According to the characteristics of the extracted features, Random Forest classifier, XGBoost classifier and Gaussian Naive Bayes classifier are designed respectively. The three classifiers are combined according to the stacking strategy to form DMF classifier, and the second layer of model is logistic regression.  
<div align="center">
<img src=https://github.com/Timeless-zfqi/AS-DMF-framework/blob/main/Figure/stacking.jpg width=50% />
</div>  
  
### Implement your own algorithm  
In DMF classifier, there is no limitation for your implementation. All you need is ensure all models have the ability to output probability. Among them {pipe1, pipe2, pipe3, meta_classifier}  
```python
def model(num):
    sclf = RandomForestClassifier(max_depth=12,n_estimators=100,oob_score=True,n_jobs=-1)
    sxgb = XGBClassifier(eval_metric=['logloss','auc','error'],max_depth=12,n_estimators=120,n_jobs=-1)
    sgnb = GaussianNB()
    pipe1 = make_pipeline(ColumnSelector(cols=range(num)),sclf)
    pipe2 = make_pipeline(ColumnSelector(cols=range(num)),sxgb)
    pipe3 = make_pipeline(ColumnSelector(cols=range(num)),sgnb)

    stack = StackingClassifier(classifiers=[pipe1,pipe2,pipe3], meta_classifier=LogisticRegression(solver="lbfgs"))
return stack
```  
## Query and training  
After completing the modeling, you can quickly build an AS-DMF query framework using the Toolbox tool in the ALiPy package. The framework uses a pool-based active learning approach and a specific query strategy for querying, labeling and training. You need to pre-set a labeled training set L and a large pool of unlabeled samples U. The sample size of L and U can be set by yourself.  
```python
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.001, split_count=10)
```
ALiPy provides us with diverse query strategies, or combine and design new ones according to your own needs. Take QBC adoption as an example to quickly implement a query operation.  
```python
alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.001, split_count=10)

# Use the default Logistic Regression classifier
model = alibox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 50)

# Use pre-defined strategy
QBCStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
QBC_result = []

for round in range(10):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = QBCStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    QBC_result.append(copy.deepcopy(saver))

analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QBC', method_results=QBC_result)
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=True)
```  
## Acknowledgement
Thanks for these awesome resources that were used during the development of the AS-DMF framework：  
* https://www.stratosphereips.org/datasets-ctu13
* https://www.wireshark.org/
* https://zeek.org/
* https://github.com/zeek-flowmeter/zeek-flowmeter
* https://github.com/SuperCowPowers/zat
* https://scikit-learn.org/stable/index.html

## Contact  
