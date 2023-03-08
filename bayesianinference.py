import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as nvm

# Load data from csv file
data = pd.read_csv('/users/quies/downloads/exNB.csv', header=None)

# Extract feature matrix X and target vector y
X = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()

# Display the number of unique values in y
print(np.unique(y))

# Plot histograms of X values for each value of y
plt.figure()
plt.hist(X[y==1,0], label="Male", alpha=0.5, bins=30)
plt.hist(X[y==0,0], label="Female", alpha=0.5, bins=30)
plt.legend()

# Define a Gaussian Naive Bayes classifier
class GaussNB():
    
    def fit(self, X, y, epsilon=1e-3):
        # Store likelihoods and priors for each class
        self.likelihoods = {}
        self.priors = {}

        # Get unique class labels
        self.classes = np.unique(y)

        # Calculate likelihoods and priors for each class
        for k in self.classes:
            X_k = X[y==k,:]

            # Calculate mean and covariance for each feature
            self.likelihoods[k] = {"mean": np.mean(X_k, axis=0), "cov": np.cov(X_k.T) + epsilon * np.eye(X.shape[1])}
            self.priors[k] = len(X_k) / len(X)
    
    def predict(self, X):
        # Get number of samples and number of features
        N, D = X.shape
        
        # Initialize array to store predicted probabilities for each class
        P_hat = np.zeros((N, len(self.classes)))
        
        # Calculate log-probability for each class
        for k, l in self.likelihoods.items():
            P_hat[:,k] = multivariate_normal.logpdf(X, mean=l["mean"], cov=l["cov"]) + np.log(self.priors[k])
        
        # Return index of class with highest probability
        return np.argmax(P_hat, axis=1)

def accuracy(y, y_hat): 
    return np.mean(y==y_hat)

gnb = GaussNB()
gnb.fit(X, y)
y_hat = gnb.predict(X)
#Training Accuracy accuracy(y)

print(accuracy(y, y_hat))

plt.figure(figsize=(12,8))
#plt.scatter(x[,0], X[,1], c=y, alpha=0.55, s=10)
plt.scatter(X[:,0], X[:,1], c=y_hat, alpha=0.55, s=10)
