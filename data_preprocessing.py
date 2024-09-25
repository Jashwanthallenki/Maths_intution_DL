#separating dependent and independent variables

import pandas as pd
import numpy as np

df = pd.read_csv(r"churn.csv")

X = df.drop(columns=['Churn'],axis=1,inplace=True)

y = df['Churn']

y = np.where(y.str.contains("No"),0,1)


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

preprocess = LabelEncoder()
SS = StandardScaler()

# Loop through each column in the DataFrame X
for i in X.columns:
    if X[i].dtype == 'object':  # Check if the column is categorical (object type)
        X[i] = preprocess.fit_transform(X[i])
    elif np.issubdtype(X[i].dtype, np.number):  # Check if the column is numeric
        X[i] = SS.fit_transform(X[i].values.reshape(-1, 1))  # Standardize the numeric column

# Now X has encoded categorical columns and standardized numerical columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
y.reshape(1,-1)

#initial parameters
def initial(input_size,hidden_size,output_size):
    np.random.seed(42)

    W1 = np.random.randn(input_size,hidden_size)*0.01
    b1 = np.zeros((1,hidden_size))
    W2 = np.random.randn(hidden_size,output_size)*0.01
    b2 = np.zeros((1,output_size))

    return W1,b1,W2,b2

#Activation Function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return z*(1-z)

#Forward propagation
def forward_propagation(X,W1,b1,W2,b2):

    Z1 = np.dot(X.T,W1) + b1

    A1 = sigmoid(Z1)

    Z2 = np.dot(A1,W1.T) + b2

    A2 = sigmoid(Z2)

    return Z1,A1,Z2,A2


#cost function
def cost(A2,y):
    loss = -np.mean(y*np.log(A2) + (1-y)*np.log(1-A2))
    return loss



