# Churn Prediction Model

A TensorFlow model to predict whether or not customers will leave the bank.

## The dataset

This model is based off the [Churn Prediction using Machine Learning on Kaggle](https://www.kaggle.com/code/simgeerek/churn-prediction-using-machine-learning/input)

The dataset has the following columns (pulled from Kaggle description):

- RowNumber—corresponds to the record (row) number and has no effect on the output.
- CustomerId—contains random values and has no effect on customer leaving the bank.
- Surname—the surname of a customer has no impact on their decision to leave the bank.
- CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
- Geography—a customer’s location can affect their decision to leave the bank.
- Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
- Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
- Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
- Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
- NumOfProducts—refers to the number of products that a customer has purchased through the bank.
- HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
- IsActiveMember—active customers are less likely to leave the bank.
- EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
- Exited—whether or not the customer left the bank.

## Implementation

### Data Preprocessing

First, unecessary features are removed from the input data such as `CustomerId`, `Surname`, and `RowNumber`. The labels are extracted from the `Exited` column.

A `LabelEncoder` is then applied to categorical data.

The data is then split into **training**, **testing**, and **validation** subsets with a 75%-15%-10% ratio respectively.

Finally, the data is standardized using `sklearn.preprocessing.StandardScaler`

### Network architecture

The network is very simple and consists of only 1 hidden dense layer:

- Dense layer with **128 neurons** and **ReLu activation**
- Dense output layer with **1 neuron** and **Sigmoid activation**

The sigmoid activation is used since this is a _binary classification problem_. The output is either churn or not churn.

### Training

The model was trained using the Adam optimizer with binary crossentropy loss. In 10 epochs with a batch size of 10, the model had an 86% accuracy rate on the testing data.
