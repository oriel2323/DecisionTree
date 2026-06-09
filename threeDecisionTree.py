#Oriel Borgharkar - 337732739

#The dataset
from sklearn.datasets import load_iris, load_wine, load_digits 
#With this we can divide the training part and the testing part
from sklearn.model_selection import train_test_split
#For building decision tree
from sklearn.tree import DecisionTreeClassifier
# For these values calculation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# helps for creating a graph
import matplotlib.pyplot as plt

def analyze_dataset(dataset,name):   
    #on which dataset we are working
    print("\nDataset:", name)
    
    #X is the attributes of the flowers
    X = dataset.data
    # y is the label
    y = dataset.target

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

    print(" ")
    print("Depth | Accuracy | Precision | Recall | F1-score")
    print("-----------------------------------------------")

    # Lists for saving the results of each tree depth
    depths_list = []  
    accuracies_list = []
    precisions_list = []
    recalls_list = []
    f1_scores_list = []

    # we do the same things, but this time we added max_depth 
    # The for loop will go over every iteration and calculate accuracy,precision,recall and f1_score
    for depth in range(1,treeDepth+1):

        # Create a temporary decision tree with the current depth
        tempTree = DecisionTreeClassifier(criterion="entropy", max_depth = depth, random_state=7)

        # Train the temporary tree using the training data
        tempTree.fit(XTrain,yTrain)
        # Predict the labels of the test data
        yPred = tempTree.predict(XTest)

        # Calculate the evaluation metrics for the current tree
        # zero_division=0 prevents warnings when a class was not predicted
        accuracy = accuracy_score(yTest,yPred)
        precision = precision_score(yTest,yPred,average = "macro", zero_division = 0)
        recall = recall_score(yTest, yPred, average="macro", zero_division = 0)
        f1 = f1_score(yTest, yPred, average="macro", zero_division = 0)

        # Print the results in a table format with 4 numbers after the point. (using fstring)
        print(f"{depth:<5} | {accuracy:<8.4f} | {precision:<9.4f} | {recall:<8.4f} | {f1:<8.4f}")

        # Save the results in lists for the graph
        depths_list.append(depth)
        accuracies_list.append(accuracy)
        precisions_list.append(precision)
        recalls_list.append(recall)
        f1_scores_list.append(f1)

    # Find the best accuracy value
    bestAccuracy = max(accuracies_list)
    # Find the index of the best accuracy
    bestIndex = accuracies_list.index(bestAccuracy)
    # Find the tree depth that gave the best accuracy
    bestDepth = depths_list[bestIndex]

    # Print the best depth and its accuracy
    print("The best Depth for ", name, ":", bestDepth)
    print("The best Accuracy for ",name, ":", bestAccuracy)

    # Create a new figure for the current dataset graph
    plt.figure()
    # Add a title to the graph with the dataset name
    plt.title(name + " - Decision Tree Metrics")

    # Draw the accuracy line for each tree depth
    plt.plot(depths_list, accuracies_list, label='accuracy')

    # Draw the precision line for each tree depth
    plt.plot(depths_list, precisions_list, label='precision')

    # Draw the recall line for each tree depth
    plt.plot(depths_list, recalls_list, label='recall')

    # Draw the F1-score line for each tree depth
    plt.plot(depths_list, f1_scores_list, label='f1-score')

    # Label the x-axis and y-axis as Tree depth and Score
    plt.xlabel('Tree Depth')                      
    plt.ylabel('Score')                           

    # Show the labels of the lines on the graph
    plt.legend()                                  

    # Display the graph
    plt.show()


analyze_dataset(load_iris(),"iris")
analyze_dataset(load_wine(),"wine")
analyze_dataset(load_digits(),"digits")






