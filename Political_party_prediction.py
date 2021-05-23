
import pandas as pd 
#from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt
# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv("house-votes-84.data.txt"  , 
    sep= ',', header = None) 
    majority=[]
    for i in range(1,17):
        c1=0
        c2=0
        for j in range(0,435):
            if balance_data[i][j]=="y":
                c1+=1
            elif balance_data[i][j]=="n":
                c2+=1
        if c1>=c2:
            majority.append('y')
        else:
          majority.append('n') 
    #print(majority)
    for i in range(1,17):
        for j in range(0,435):
            if balance_data[i][j]=="?":
                balance_data[i][j]=majority[i-1]
    
    #print(balance_data)
    return balance_data 
    
# Function to split the dataset 
def splitdataset(balance_data,per): 
    for i in range(1,17):
        for j in range(0,435):
            if balance_data[i][j]=="n":
                balance_data[i][j]=0
            else:
                balance_data[i][j]=1
    # Separating the target variable 
    X = balance_data.values[:,1:17] 
    Y = balance_data.values[:,0] 
    
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 1-per, random_state = None) 
    return X, Y, X_train, X_test, y_train, y_test


      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = None, 
            max_depth = None, min_samples_leaf = 17) 
    clf_entropy.fit(X_train, y_train) 
    
    return clf_entropy 
def prediction(X_test, clf_object): 
  
    y_pred = clf_object.predict(X_test)  
    return y_pred 

def cal_accuracy(y_test, y_pred): 
      
     
    acc=accuracy_score(y_test,y_pred)*100
    return acc
def cal_all_function(per ,data ):
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data,per)
    
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
    size = clf_entropy.tree_.node_count
    y_pred_entropy = prediction(X_test, clf_entropy) 
    acc=cal_accuracy(y_test, y_pred_entropy) 
    return size , acc,len(X_train)

# Driver code 
def main():  
    # Building Phase 
    print('accuracy' , '             ' , 'size')
    for i in range(1,6):
        per = 0.25
        data = importdata()
        size , acc , Len =cal_all_function(per , data)
        print(acc , '     ', size)
        
    print('/n')
    print('min_Saize' , '   ' , 'Max_size', '   ' ,'avarge_size', '   ' ,'min_accuracy' , '         ' , 'Max_accuracy', '          ' ,'avarge_accuracy', '          ','traning size' )
    per=0.3
    trsze=[]
    accuracy=[]
    noonodes=[]
    avar_size = []
    avar_acc = []
    
    for i in range(1,6):
        for j in range(1,6):
            data = importdata()
            size , acc,Len =cal_all_function(per , data)
            accuracy.append(acc)
            noonodes.append(size)
        trsze.append(Len)
        smin = min(noonodes)
        smax = max(noonodes)
        savar = sum(noonodes) / len(noonodes)
        avar_size.append(savar)
        accmin = min(accuracy)
        accmax = max(accuracy)
        accavar = sum(accuracy) / len(accuracy)
        avar_acc.append(accavar)
        print(smin,'            ',smax,'            ',savar ,'        ',accmin ,'      ',accmax,'       ', accavar,'      ',Len)
        per+=0.1
   # print(avar_acc,avar_size,trsze)
    plt.subplot(2, 1,1)
    plt.plot(trsze,avar_acc)
    plt.xlabel('training set size')
    plt.ylabel('accuracy')
    plt.show()
    plt.subplot(2, 1, 2)
    plt.plot(trsze,avar_size)
    plt.xlabel('training set size')
    plt.ylabel('numberofnodes')
    plt.show()
    
   
# Calling main function 
if __name__=="__main__": 
    main() 