class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=100000, intercept=True,):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.intercept = intercept

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):

        X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            loss = self.loss(h, y)


    def predict_prob(self, X):
        X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()

model = LogisticRegression(learning_rate=0.1, num_iter=300000)
