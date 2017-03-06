       
        LR = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', tol=10**-7, max_iter=1000)
        LR.fit(X, C)
        for c in C:
            print c,
        self.backend = LR
        

        #Y = self.backend.predict(X_to_predict)