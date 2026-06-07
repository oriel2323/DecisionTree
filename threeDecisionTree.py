#Oriel Borgharkar - 337732739

#The dataset
from sklearn.datasets import load_iris
#With this we can divide the training part and the testing part
from sklearn.model_selection import train_test_split
#For building decision tree
from sklearn.tree import DecisionTreeClassifier
# For these values calculation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#load the data
iris = load_iris() 

#X is the attributes of the flowers
X = iris.data
# y is the label
y = iris.target

#90% of the data for training and 10% for testing
XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size = 0.1, random_state = 7, stratify = y) 

