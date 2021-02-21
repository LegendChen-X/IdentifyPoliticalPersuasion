#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    if C.sum(): return C.trace() / C.sum()
    return 0.0


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    res = []
    C = C.T
    for i in range(C.shape[0]):
        if C[i].sum(): res.append(C[i][i] / C[i].sum())
        else: res.append(0.0)
    return res
            

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    res = []
    for i in range(C.shape[0]):
        if C[i].sum(): res.append(C[i][i] / C[i].sum())
        else: res.append(0.0)
    return res
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    models = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10, random_state=401), MLPClassifier(alpha=0.05), AdaBoostClassifier(random_state=401)]
    outputs = {}
    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        confusionMatrix = confusion_matrix(y_test, predictions)
        outputs[str(model).split("(")[0]] = [confusionMatrix, accuracy(confusionMatrix), recall(confusionMatrix), precision(confusionMatrix)]
    iBest = 0
    best = 0.0
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        #     outf.write(f'Results for {classifier_name}:\n')  # Classifier name
        #     outf.write(f'\tAccuracy: {accuracy:.4f}\n')
        #     outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
        #     outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
        #     outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
        for model in outputs:
            outf.write(f'Results for {model}:\n')
            outf.write(f'\tAccuracy: {outputs[model][1]:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in outputs[model][2]]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in outputs[model][3]]}\n')
            outf.write(f'\tConfusion Matrix: \n{outputs[model][0]}\n\n')
            
            if outputs[model][1] > best:
                best = outputs[model][1]
                iBest = list(outputs.keys()).index(model)
    
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    sizes = [1000, 5000, 10000, 15000, 20000]
    accs = []
    bestModel = None
    
    if iBest == 0: bestModel = SGDClassifier()
    elif iBest == 1: bestModel = GaussianNB()
    elif iBest == 2: bestModel = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
    elif iBest == 3: bestModel = MLPClassifier(alpha=0.05)
    elif iBest == 4: bestModel = AdaBoostClassifier(random_state=401)
    for size in sizes:
        bestModel.fit(X_train[:size], y_train[:size])
        prediction = bestModel.predict(X_test)
        acc = accuracy(confusion_matrix(y_test, prediction))
        accs.append(acc)
            
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))
        for i in range(len(sizes)):
            outf.write(f'{sizes[i]}: {accs[i]:.4f}\n')
        outf.write("According to my output, the number of training sample has a positive impact to the"
                    "accuracy. Intuitively, more training data can absolutly increase the accuracy of our model, since our model will have more data and avoid the problem of underfit. We can see initially the increase is significant and we increase the training number from 15000 to 20000, it has low improvement. I think this is beacuse data has a high variance and our model has achieve its convergence line. We may need more complicated architecture to capture the hidden features."
                    )
            
    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    
    bestModel = None
    
    if i == 0: bestModel = SGDClassifier()
    elif i == 1: bestModel = GaussianNB()
    elif i == 2: bestModel = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
    elif i == 3: bestModel = MLPClassifier(alpha=0.05)
    elif i == 4: bestModel = AdaBoostClassifier(random_state=401)
    
    kPP = {}
    
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
            # outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        
        # outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        # outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        # outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # outf.write(f'Top-5 at higher: {top_5}\n')
        for k in [5, 50]:
            selector = SelectKBest(f_classif, k=k)
            X_new = selector.fit_transform(X_train, y_train)
            pp = selector.pvalues_
            kPP[k] = pp
            
        selector_1k = SelectKBest(f_classif, k=5)
        X_new = selector_1k.fit_transform(X_1k, y_1k)
        X_new_test = selector_1k.transform(X_test)
        bestModel.fit(X_new, y_1k)
        prediction = bestModel.predict(X_new_test)
        accuracy_1k = accuracy(confusion_matrix(y_test, prediction))
        
        selector_32k = SelectKBest(f_classif, k=5)
        X_new = selector_32k.fit_transform(X_train, y_train)
        X_new_test = selector_32k.transform(X_test)
        prediction = bestModel.predict(X_new_test)
        accuracy_full = accuracy(confusion_matrix(y_test, prediction))
        
        pp_1k = np.array(selector_1k.pvalues_)
        pp_32k = np.array(selector_32k.pvalues_)
        
        indices_1k = np.argpartition(pp_1k, 5)
        indices_32k = np.argpartition(pp_32k, 5)
        feature_intersection = np.intersect1d(indices_1k[:5], indices_32k[:5])
        
        interList = []
        for element in feature_intersection: interList.append(element)
        
        for k in [5, 50]:
            outf.write(f'{k} p-values: {[round(pval, 4) for pval in kPP[k]]}\n')
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {interList}\n')
        outf.write(f'Top-5 at higher: {indices_32k[:5].tolist()}\n')
        outf.write("The top 5 features for 1K set and 32K set are the same according to my output. They are  Number of first-person pronouns, Number of second-person pronouns, Number of adverbs and two from LIWC/Receptiviti features. I think the number of first-person pronouns can help us identify the subjective level of one comment and the number of second-person pronouns can help us identify the interactive level of one comment. Follow the intuition, feaures from LIWC/Receptiviti are useful to determine the semantic of that text. Since for each division, it will have its own way in semantic meaning. Pvalues are generally the same. If we round it to 4 decimals, it has no difference. Follow my intuition, a low p-value means that specific feature separate the dataset. If we have a large dataset, it will be really hard to separate. Therefore, we should have a high p value")

def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    kFlods = KFold(n_splits=5, shuffle=True, random_state=401)
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test), axis=0)
    accs = {}
    for j in range(5): accs[j] = []
    for train, test in kFlods.split(X):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        model = None
        for j in range(5):
            if j == 0: model = SGDClassifier()
            elif j == 1: model = GaussianNB()
            elif j == 2: model = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
            elif j == 3: model = MLPClassifier(alpha=0.05)
            elif j == 4: model = AdaBoostClassifier(random_state=401)
            
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            acc = accuracy(confusion_matrix(y_test, prediction))
            accs[j].append(acc)
            
    pVals = []
    for key in accs:
        if key != i: pVals.append(ttest_rel(accs[key], accs[i]).pvalue)
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        for j in range(5):
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in accs[j]]}\n')
        outf.write(f'p-values: {[round(pVal, 4) for pVal in pVals]}\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
    inputData = np.load(args.input)
    data = inputData[inputData.files[0]]
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], train_size=0.8, random_state=401, shuffle=True)
    print("3.1 begin")
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    print("3.1 finish")
    print("3.2 begin")
    (X_1k, y_1k) = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    print("3.2 finish")
    print("3.3 begin")
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    print("3.3 finish")
    print("3.4 begin")
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    print("3.4 finish")
    
