"""
@Author: Sai Yadavalli
Version: 1.2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression:
    
    def __init__(self, df, yCol):
        """
        Initializes the LogisticRegression class.

        Args:
            df (pd.DataFrame): The dataset to be used for training.
            yCol (str): The column name representing the target variable.
        """
        self.df = df
        self.yCol = yCol
        self.m = len(df.index)
        self.n = len(df.columns)
        self.X, self.y = self.ArrayMaker(df, yCol)
        self.W = np.zeros((self.n, 1))

    @staticmethod
    def hwX(w, X):
        """
        Logistic regression hypothesis function.

        Args:
            w (np.ndarray): Weight vector.
            X (np.ndarray): Feature matrix.

        Returns:
            float: The hypothesis value for the given weights and features.
        """
        hwx = 1 / (1 + np.exp(np.dot(-w.T, X.T)))  # Transpose X to align shapes
        return hwx.item()  # Extract scalar value

    @staticmethod
    def Cost(w, X, y):
        """
        Computes the cost for a single training example.

        Args:
            w (np.ndarray): Weight vector.
            X (np.ndarray): Feature vector for a single example.
            y (int): Target value for the example.

        Returns:
            float: The cost for the given example.
        """
        if y == 1:
            return -np.log(LogisticRegression.hwX(w, X))
        elif y == 0:
            return -np.log(1 - LogisticRegression.hwX(w, X))

    def J(self, w, X, y, m):
        """
        Computes the average cost over all training data.

        Args:
            w (np.ndarray): Weight vector.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            m (int): Number of training examples.

        Returns:
            float: The average cost.
        """
        J = 0
        for i in range(m):
            J += self.Cost(w, X[i], y[i])
        return J / m

    def GD(self, w, X, y, m, n, alpha):
        """
        Performs gradient descent to update weights.

        Args:
            w (np.ndarray): Weight vector.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            m (int): Number of training examples.
            n (int): Number of features.
            alpha (np.ndarray): Learning rate for each feature.

        Returns:
            np.ndarray: Updated weight vector.
        """
        sum = np.zeros(n)
        new_ws = np.zeros(n)
        for i in range(m):
            for j in range(n):
                sum[j] += (self.hwX(w, X[i]) - y[i]) * X[i][j]

        for j in range(n):
            new_ws[j] = w[j] - alpha[j] * (1 / m) * sum[j]

        return new_ws.reshape(-1, 1)  # Ensure the shape matches w

    @staticmethod
    def ArrayMaker(df, yCol):
        """
        Creates feature matrix X and target vector y from a DataFrame.

        Args:
            df (pd.DataFrame): The dataset.
            yCol (str): The column name representing the target variable.

        Returns:
            tuple: Feature matrix X and target vector y.
        """
        X = df.drop(yCol, axis=1)
        X = np.insert(X.to_numpy(), 0, 1, axis=1)  # Add bias term
        y = df[yCol].to_numpy()
        return X, y

    @staticmethod
    def standardize(df, label):
        """
        Standardizes the features of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be standardized.
            label (str): The column to be classified.

        Returns:
            pd.DataFrame: The standardized DataFrame.
        """
        df_copy = df.copy(deep=True)
        for feature in df_copy.columns:
            if feature != label:
                mean = df_copy[feature].mean()
                std = df_copy[feature].std(ddof=0)
                df_copy[feature] = (df_copy[feature] - mean) / std
        return df_copy

    def Train(self, iterations, alpha, standard=False):
        """
        Trains the logistic regression model using gradient descent.

        Args:
            iterations (int): Number of iterations for gradient descent.
            alpha (np.ndarray): Learning rate for each feature.
            standard (bool): Whether to standardize the data before training.
        """
        if standard:
            self.df = self.standardize(self.df, self.yCol)
            self.X, self.y = self.ArrayMaker(self.df, self.yCol)

        w = np.zeros((self.n, 1))
        print("Initial Error (J)=", self.J(w, self.X, self.y, self.m))
        errors = []
        for k in range(iterations):
            error = self.J(w, self.X, self.y, self.m)
            errors.append(error)
            w = self.GD(w, self.X, self.y, self.m, self.n, alpha)
            print("Iteration", k, "Error (J)=", error)
        self.W = w
        print("Final Error (J)=", self.J(w, self.X, self.y, self.m))
        print("The weights:")
        print(w)
        
        # Plot only the last 100 iterations
        if iterations > 100:
            plt.plot(range(iterations - 100, iterations), errors[-100:])
        else:
            plt.plot(range(iterations), errors)
        
        plt.xlabel("Iterations")
        plt.ylabel("Error (J)")
        plt.title("Error vs. Last 100 Iterations")

    def evaluate(self, positive_class, test, train, standard=False):
        """
        Evaluates the logistic regression model using a test dataset.

        Args:
            positive_class (int): The positive class to be used.
            test (str): Path to the test dataset.
            train (str): Path to the training dataset.
            standard (bool): Whether to standardize the data before evaluation.

        Returns:
            list: A list containing the accuracy, precision, recall, and F1 score (in that order).
        """
        df = pd.read_csv(train)
        testdf = pd.read_csv(test)

        if standard:
            df = self.standardize(df, self.yCol)
            testdf = self.standardize(testdf, self.yCol)

        train_features = []
        for column in df.columns:
            if column != self.yCol:
                train_features.append(column)

        n = len(df.columns) - 1
        testm = len(testdf)
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(testm):
            # Create the test sample
            test_sample = np.zeros((1, n + 1))
            test_sample[0][0] = 1
            for j in range(n):
                test_sample[0][j + 1] = testdf.loc[i].at[train_features[j]]
            # Calculate the hypothesis
            ans = self.hwX(self.W, test_sample)
            # Convert the hypothesis to a class label
            if ans >= 0.5:
                ans = positive_class
            else:
                ans = 1 - positive_class

            if ans == testdf.loc[i].at[self.yCol] and ans == positive_class:
                TP += 1
            elif ans == testdf.loc[i].at[self.yCol]:
                TN += 1
            elif ans == positive_class:
                FP += 1
            else:
                FN += 1

        print("TP:", TP)
        print("FP:", FP)
        print("TN:", TN)
        print("FN:", FN)
        Accuracy = (TP + TN) / testm
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 1 / ((1 / Precision) + (1 / Recall))
        Errors = FN + FP

        print("Accuracy:", Accuracy)
        print("Precision:", Precision)
        print("Recall:", Recall)
        print("F1:", F1)

        return [Accuracy, Precision, Recall, F1, Errors]


#MAIN
# Create the training data
fin = input("Enter a training file: ")
df = pd.read_csv(fin)
print(df.columns)
yCol = input("Enter the column name for classes: ")
model = LogisticRegression(df, yCol)

# TRAIN THE SYSTEM
n = model.n
iterations = 5000
alpha = np.zeros(n)
for i in range(n):
    alpha[i] = 0.000009

standardize_data = input("Standardize the data? (yes/no): ").strip().lower() == "yes"
model.Train(iterations, alpha, standard=standardize_data)
plt.show()

test = input("Enter a test file: ")
model.evaluate(1, test, fin, standard=standardize_data)

