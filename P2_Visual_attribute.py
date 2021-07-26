# CS5014 (Machine Learning), Visual Attributes assignment. 
# Ashraf Sinan, University of St Andrews
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, plot_confusion_matrix, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


training_path = 'data/data_train.csv'
test_path = 'data/data_test.csv'

##Create Evaluation data frame to save the result of different experiments
evaluation_columns = ['Problem', 'Model', 'IsDatCorrect', 'Training_Accuracy', 'Testing_accuracy',
                      'Balanced_Training_Accuracy', 'Balanced_Testing_accuracy']

evaluation_data_frame = pd.DataFrame(columns=evaluation_columns)

evaluation_data_frame= pd.DataFrame(columns=evaluation_columns)

##Create many models to build three experiments

def runLogisticModel(x,y):
    ## Prepare the model hyper parameters
    multi_class='auto'
    solver='liblinear'
    max_iter=1000
    class_weight='balanced'
    
    #Create the model
    LR = LogisticRegression(multi_class= multi_class, solver=solver, 
                            max_iter=max_iter, class_weight=class_weight)
    
    # Train the model
    LR = LR.fit(x, y)

    return LR

def runSvmModel(x,y):
   
    C=0.1
    kernel = 'rbf'
    gamma= 'auto'
    class_weight='balanced'
    decision_function_shape= 'ovr'
    tol = 1e-3
    
    #Create the model
    SVM = svm.SVC(C=C, kernel = kernel,
                  gamma= gamma, class_weight=class_weight, 
                  decision_function_shape= decision_function_shape, tol=tol)
    
    # Train the model
    SVM = SVM.fit(x,y)  
    
    return SVM

def runRandomForestModel(x,y):
    ## Prepare the model hyper parameters
    n_estimators=50
    max_leaf_nodes=20
    min_samples_split=0.01
    n_jobs=-1
    class_weight='balanced'
    
    RFC = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,   
                                 min_samples_split=min_samples_split, n_jobs=n_jobs, 
                                 class_weight='balanced')
    # Train the model
    RFC =   RFC.fit(x,y)
    
    return RFC
    

def runDecisionTreeModel(x,y):
    #Prepare the model hyper parameters
    max_depth=15
    min_impurity_decrease=0.01
    #Create the model
    DTC = DecisionTreeClassifier(max_depth=max_depth, 
                                 min_impurity_decrease=min_impurity_decrease)
    
    #Train the model
    DTC = DTC.fit(x,y)
    
    return DTC


def runNNModel(x,y):
    #Prepare the model hyper parameters
    solver='adam'
    alpha=1e-7
    hidden_layer_sizes=(500,500,500)
    activation='relu'
    validation_fraction=0.2
    max_iter=100
    
    #Create the model and train it
    NN = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation, validation_fraction=validation_fraction, 
                       max_iter=max_iter)
    NN = NN.fit(x, y)
   
    return NN

def runExtraTreeModel(x,y):
    
    max_depth=10
    min_impurity_decrease=0.001
    class_weight='balanced' 
    
    ETM = ExtraTreeClassifier(max_depth=max_depth, 
                              min_impurity_decrease=min_impurity_decrease,  
                              class_weight=class_weight)
    
    ETM = ETM.fit(x,y)
    
    return ETM

def loadData(dataPath):
    data = pd.read_csv(dataPath)
    return data


def cleanData(data):
    cleanData = data.dropna(axis=0)
    return cleanData


def getColorData(data):
    #Extract the CIELAB features and colour
    x = data.loc[:,'lightness_0_0':'blueyellow_2_2']
    y = data.loc[:,['color']]
    #Encode the class and save the mapping dictionary to map it back later
    color_data_cat = y.astype('category')
    color_data_codes = color_data_cat['color'].cat.codes
    color_data_dict = color_data_cat['color'].cat.categories
  
    return x, color_data_codes, color_data_dict

def getTextureData(data):
    ##Extract the texture predictors and texture
    x = data.loc[:,'hog_0_0_0':]
    y = data.loc[:,['texture']]
    #Encode the class and save the mapping dictionary to map it back later
    texture_data_cat = y.astype('category')
    texture_data_codes = texture_data_cat['texture'].cat.codes
    texture_data_dict = texture_data_cat['texture'].cat.categories
  
    return x, texture_data_codes, texture_data_dict

def mapData(data, dictionary, column):
    ## Map the classes back after prediction
    mappedData = data[column].map(dictionary)
    
    return mappedData
    


def evaluatePerformance(model, x_train, x_test, y_train, y_test, problem, modelName, data_correctness):
    global evaluation_data_frame
    
    ## Get training accuracy and testing accuracy, balanced and unbalanced
    training_pred = model.predict(x_train)
    testing_pred = model.predict(x_test)
    
    training_accuracy = accuracy_score(training_pred, y_train)
    training_balanced_accuracy = balanced_accuracy_score(training_pred, y_train)
    testing_accuracy = accuracy_score(testing_pred, y_test)
    testing_balanced_accuracy = balanced_accuracy_score(testing_pred, y_test)
    
    ## Create a record to add to the evaulation data frame
    result_record = [problem, modelName, data_correctness, training_accuracy, testing_accuracy, 
                    training_balanced_accuracy, testing_balanced_accuracy]
    series = pd.Series(result_record, index = evaluation_columns)
    
    ## Append the record to the datafame to conduct data analysis later
    evaluation_data_frame = evaluation_data_frame.append(series, ignore_index=True)
    


def visualizeColor(data, color_data):
    # Set the plotting parameters
    fontsize = "20";
    params = {'figure.autolayout':True,
              'legend.fontsize': fontsize,
              'figure.figsize': (20,20),
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
               'xtick.labelsize':fontsize,
              'ytick.labelsize':fontsize}
   
    col_names = ['lightness', 'redgreen','blueyellow']
    #Create a new data frame to separate the 9 areas of the object
    x = data.loc[:,'lightness_0_0':'blueyellow_2_2']
    
    new_data = pd.DataFrame(columns=col_names)
    
    #split each area of the 9 areas to a different class taking the same color of the main object
    for i in range(3):
        for j in range(3):
            temp_df =  pd.DataFrame()
            color_index = '_' + str(i) + '_' + str(j)
            for col in col_names:
                temp_df[str(col)] = x[col + color_index]
            temp_df['color'] = color_data
            new_data = new_data.append(temp_df)

    #Prpeare a  3D plot
    
    plt.rcParams.update(params)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    #Plot the data and set labels
    ax.scatter(new_data['lightness'], new_data['redgreen'],  new_data['blueyellow'], c=new_data["color"])
    ax.set_xlabel('Red green')
    ax.set_ylabel('Blue yellow')
    ax.set_zlabel('Lightness')
    ax.set_title("Color distribution:")
    plt.show()
    fig.savefig("fig.png")

#Load data and clean it
training_data = loadData(training_path)
clean_data = cleanData(training_data)

## separate color and texture data
x_color, y_color, color_dictionary = getColorData(clean_data)
x_texture, y_texture, texture_dictionary = getTextureData(clean_data)

## Split the data to training and validation

x_color_train, x_color_validate, y_color_train, y_color_validate = train_test_split(x_color, y_color,
                                                                                    stratify = y_color,
                                                                                    test_size=0.2)
x_texture_train, x_texture_validate, y_texture_train, y_texture_validate = train_test_split(x_texture, 
                                                                                            y_texture, 
                                                                                            stratify = y_texture,
                                                                                            test_size=0.2)
## Scale the continuous features
x_color_train, x_color_validate, x_texture_train, x_texture_validate = (scale(x_color_train), 
                                                                        scale(x_color_validate), 
                                                                        scale(x_texture_train), 
                                                                        scale(x_texture_validate))
## Visualize the colours
visualizeColor(clean_data, clean_data.loc[:,'color'])

#Save models in a List of function to call all of them using a loop
models = [runDecisionTreeModel, runExtraTreeModel, runLogisticModel, 
          runNNModel, runRandomForestModel, runSvmModel]

models_names = ["Decision Tree", "Extra Tree", "Logistic Regression", 
                "Neural Netwrok", "Random Forest", 'SVMS']

##Create a function to call all models and once for experiments reasons

def testSixModels(x_train, x_test, y_train, y_test, problem, data_correctness):
    ## Get the data from global variable above
    global models_names
    
    ## Try all models on the models list on the problem and store the evaluation to the evaluation dataframe
    for i in range(len(models)):
        model_name = models_names[i]
        model = models[i](x_train, y_train)
        evaluatePerformance(model, x_train, x_test, y_train, y_test, problem, model_name, data_correctness)




##Experiment 1 test all models on correct features for color and texture
data_correctness = 'correct'
problem_name = 'color'
testSixModels(x_color_train, x_color_validate, y_color_train, y_color_validate, 
              problem_name, data_correctness)
problem_name = 'texture'
testSixModels(x_texture_train, x_texture_validate, y_texture_train, y_texture_validate,
              problem_name, data_correctness)


#Expirement 2 (split the data in a wrong way),  we will use colour predictor to predict texture and vice versa
x_color_train, x_color_validate, y_color_train, y_color_validate = train_test_split(x_texture, y_color,
                                                                                    stratify = y_color, 
                                                                                    test_size=0.2)
x_texture_train, x_texture_validate, y_texture_train, y_texture_validate = train_test_split(x_color , 
                                                                                            y_texture,
                                                                                            stratify = y_texture,
                                                                                            test_size=0.2)

## Scale the data
x_color_train, x_color_validate, x_texture_train, x_texture_validate = (scale(x_color_train), 
                                                                        scale(x_color_validate), 
                                                                        scale(x_texture_train), 
                                                                        scale(x_texture_validate))

## Try the models on colour 
data_correctness = 'wrong'
problem_name = 'color'
testSixModels(x_color_train, x_color_validate, y_color_train, y_color_validate, 
              problem_name, data_correctness)
problem_name = 'texture'
testSixModels(x_texture_train, x_texture_validate, y_texture_train, y_texture_validate,
              problem_name, data_correctness)

## Expirment 3 Try the classifycation models using random dataset to predict the outputs
#Create random data for colour and texture
x_color_random = np.random.uniform(low=0.0, high=1.0, size=x_color.values.shape)
x_texture_random = np.random.uniform(low=0.0, high=1.0, size=x_texture.values.shape) 

x_color_train, x_color_validate, y_color_train, y_color_validate = train_test_split(x_color_random, y_color,
                                                                                    stratify = y_color,
                                                                                    test_size=0.2)
x_texture_train, x_texture_validate, y_texture_train, y_texture_validate = train_test_split(x_texture_random , y_texture,
                                                                                            stratify = y_texture,
                                                                                            test_size=0.2)
## Try the models on the random data for both texture and color
data_correctness = 'Random'
problem_name = 'color'
testSixModels(x_color_train, x_color_validate, y_color_train, y_color_validate, 
              problem_name, data_correctness)
problem_name = 'texture'
testSixModels(x_texture_train, x_texture_validate, y_texture_train, y_texture_validate,
              problem_name, data_correctness)

print(evaluation_data_frame)
## Export the evaluation result to CSV
evaluation_data_frame.to_csv("Model Testing results.csv")