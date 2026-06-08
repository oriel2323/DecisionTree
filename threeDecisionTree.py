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

# Create a decision tree classifier using entropy
tree = DecisionTreeClassifier(criterion= "entropy" , random_state = 7)

# Train the decision tree using the training data
tree.fit(XTrain,yTrain)

# Get the depth of the trained decision tree
treeDepth = tree.get_depth()
print("Tree depth: ", treeDepth)

#Predict the labels for the test data
yPred = tree.predict(XTest)

# Calculate evaluation metrics
#we use macro average because in this data set the catagories are not only 1 or 0 
# thats why we need to do average. 
accuracy = accuracy_score(yTest,yPred)
precision = precision_score(yTest,yPred,average = "macro")
recall = recall_score(yTest, yPred, average="macro")
f1 = f1_score(yTest, yPred, average="macro")

#print the values
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)


print("Depth | Accuracy | Precision | Recall | F1-score")
print("-----------------------------------------------")


# we do the same things, but this time we added max_depth 
# The for loop will go over every iteration and calculate accuracy,precision,recall and f1_score
for depth in range(1,treeDepth+1):
    tempTree = DecisionTreeClassifier(criterion="entropy", max_depth = depth, random_state=7)
    tempTree.fit(XTrain,yTrain)

    yPred = tempTree.predict(XTest)

    accuracy = accuracy_score(yTest,yPred)
    precision = precision_score(yTest,yPred,average = "macro")
    recall = recall_score(yTest, yPred, average="macro")
    f1 = f1_score(yTest, yPred, average="macro")    
    print(f"{depth:<5} | {accuracy:<8} | {precision:<9} | {recall:<6} | {f1:<8}")




