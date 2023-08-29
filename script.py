#IDAML PROJECT
#letters spam vs non-spam
#bow representation with vocab 75173 
#max 0.2% misrepresent spam emails
#10k mails training:
#Y == +1 for spam 
#Y == -1 for non-spam
import scipy.io#read matlab
#from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid,train_test_split
from sklearn.metrics import PrecisionRecallDisplay,confusion_matrix,f1_score,precision_recall_curve
import matplotlib.pyplot as plt
import math,os
import numpy as np
import pandas as pd

X_train = None
X_val = None
X_test = None
y_train = None
y_val = None
y_test = None


#Preprocessing with this huge matrix is SLOW
#better have some save and load functions
def save_data(step='split'):
    if step == 'split':
        X_train.sparse.to_dense().to_parquet(f'X_train_{step}.parquet')
        X_val.sparse.to_dense().to_parquet(f'X_val_{step}.parquet')
        X_test.sparse.to_dense().to_parquet(f'X_test_{step}.parquet')
        y_train.to_parquet(f'y_train_{step}.parquet')
        y_val.to_parquet(f'y_val_{step}.parquet')
        y_test.to_parquet(f'y_test_{step}.parquet')
    else: 
        X_train.to_parquet(f'X_train_{step}.parquet')
        X_val.to_parquet(f'X_val_{step}.parquet')
        X_test.to_parquet(f'X_test_{step}.parquet')
        y_train.to_parquet(f'y_train_{step}.parquet')
        y_val.to_parquet(f'y_val_{step}.parquet')
        y_test.to_parquet(f'y_test_{step}.parquet')
        


def load_data(step='split'):
    global X_train,X_val,X_test,y_train,y_val,y_test
    X_train = pd.read_parquet(f'X_train_{step}.parquet')
    X_val = pd.read_parquet(f'X_val_{step}.parquet')
    X_test = pd.read_parquet(f'X_test_{step}.parquet')
    y_train = pd.read_parquet(f'y_train_{step}.parquet')
    y_val = pd.read_parquet(f'y_val_{step}.parquet')
    y_test = pd.read_parquet(f'y_test_{step}.parquet')


print('--------PREPROCESSING--------')

print('Loading data ...')
if not os.path.exists('y_test_split.parquet'):
    print('Doing Train/Test split...')
    emails = scipy.io.loadmat('emails.mat')
    #Original Format of emails['X'] is (word_id,email_num), occurrences
    X = pd.DataFrame.sparse.from_spmatrix(emails['X']).T
    #After this we have document_id,feature1,feature2.....
    #Single column 
    y = pd.DataFrame(emails['Y']).transpose()
    #After this we have document_id,class 
    seed = 123
    #RATIO TRAIN,VALIDATION,TEST=60,20,20
    #TESTING
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) 
    #TRAINING VALIDATION # 0.25 * 0.8 = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=seed) 
    save_data('split')
    print('Train/Test split is done')
else:
    print('Data split found.')
    load_data('split')

print('Loaded data.')


###We are facing a classification binary problem
print('----------------------')
print("Let's look at some statistics:")
print('----------------------')
print("On the y side:")
print("We have a heavily unbalanced dataset spam:8030,non-spam:1970")
print(y_train.value_counts())
print("The shape is [document_id,is_spam]")
print(y_train[0])
print(f"Missing values are:{y_train.isna().sum()}" )
print('----------------------')
print("On the X side:")
print("The shape is [document_id,word_id]")
print(X_train)
print("A document may have multiple occurrences of the same word and this is noted in the matrix.")
print("Max repetitions of word in a single document:",X_train.max().max())
print(f"Missing values are:{X_train.isna().sum()}" )
print('----------------------')

word_max = X_train.max().sort_values().values
print("We can see 99% of words have a very low maxim")
plt.plot(word_max)
plt.ylabel("Max occurrences per document distribution")
plt.show()

print(f"Total if we limit max to 50: {len([i for i in word_max if i< 50])}")
plt.plot([i for i in word_max if i< 50])
plt.ylabel("Max occurrences per document distribution")
plt.show()#



    
if not os.path.exists('y_test_features.parquet'):
    print('Doing feature work...')
    print('On y we will only turn -1 to 0 and then turn it into a boolean')
    print(f'Before {y_train[:5]}' )
    y_train = y_train.replace(-1.,0.).applymap(bool)
    y_val = y_val.replace(-1.,0.).applymap(bool)
    y_test = y_test.replace(-1.,0.).applymap(bool)
    print(f'After {y_train[:5]}' )
    print('On the X side we have more analysis.')
    max_val = 50
    print(f'We will lower values above {max_val} to {max_val}.')    
    X_train.where(X_train <= max_val, max_val, inplace=True)
    X_val.where(X_val <= max_val, max_val, inplace=True)
    X_test.where(X_test <= max_val, max_val, inplace=True)
    bins = 5
    print(f'Then split data into {bins} bins')
    def to_discrete(val):
        return math.ceil(float(val)/10)
    X_train = X_train.applymap(to_discrete)
    X_val = X_val.applymap(to_discrete)
    X_test = X_test.applymap(to_discrete)

    save_data('features')
    print('Train/Test split is done')
else:
    print('Data featurized found.')
    load_data('features')







print('--------TRAINING--------')
y_train= np.ravel(y_train)
y_test= np.ravel(y_test)



print('---Training Baseline tree---')
tree = DecisionTreeClassifier(criterion='gini',class_weight='balanced')
tree.fit(X_train,y_train)
y_pred = tree.predict(X_val)
tn, fp, fn, tp = confusion_matrix(y_val,y_pred).ravel()
best_f1 = f1_score(y_val,y_pred)
best_secondary_metric = fp/(fp + tn)
best_tree = tree

#We re-train the best model
print("F1: %0.5f" % best_f1) 
print(f'req1:{fp/(fp + tn)}')
print(best_tree.get_params())
print(f"The total amount of missclassified in Validation is {best_secondary_metric *100}%") 


print('---Training Main Model(Random Forest)---')
grid = {'n_estimators':[50,100,200,500],'max_depth':[50,100,200,500]}
grid = {'n_estimators':[100],'max_depth':[50]}
rfc = RandomForestClassifier(class_weight='balanced')
print('We will use f1 to choose our best model.')
print('Guaranteeing also 0.2% of all legitimate mails not being missclassfied.')
best_f1 = 0
best_grid = {'n_estimators':50,'max_depth':10}
best_rfc = None
best_secondary_metric = 500
for g in ParameterGrid(grid):
    print(f'Testing grid value:{g}')
    rfc.set_params(**g)
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val,y_pred).ravel()
    #make sure that no more than 0.2% of all legitimate emails are filtered.
    req1 = fp/(fp + tn) 
    #better f1
    req2 = f1_score(y_val,y_pred)
    #save if best and valid
    print(f'F1:{f1_score(y_val,y_pred)}')
    print(f'req1:{fp/(fp + tn)}')
    #Since I was not able to satisfy the .2%, we will prioritize the asked metric.
    if req1 < best_secondary_metric:
        if req2 > best_f1:
            best_f1 = req2
            best_secondary_metric = req1
            best_grid = g
            best_rfc = rfc

#We re-train the best model
print("Best F1: %0.5f" % best_f1) 
print("Hyperparams:", best_grid)
print(best_rfc.get_params())
print(f"The total amount of missclassified in Validation is {best_secondary_metric *100}%") 


print('--------TESTING--------')

def testing_statistics(model):
    #Make a statement about the ex-pected quality of the filter 
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    f1 = f1_score(y_test,y_pred)
    print(f'The testing F1 is: {f1}')  
    print(f"The total amount of missclassified in Testing is {fp/(fp + tn) *100}%") 

print('----Using test dataset on KNN----')
testing_statistics(best_tree)
print('----Using test dataset on RFC----')
testing_statistics(best_rfc)


print('----Plotting recall/precision curve----')

#For this purpose, plot a precision/recall curve and mark the model’s position
#on the curve (for the selected threshold). 
y_pred = best_rfc.predict(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()

