class F1History(tf.keras.callbacks.Callback):

    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):

        logs['F1_score_train'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        y_pred = (np.asarray(self.model.predict(X_train))).round()
        score = f1_score(y_train, y_pred, average='macro')

        if (self.validation):
            # logs['F1_score_val'] = float('-inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = (np.asarray(self.model.predict(X_valid))).round()
            y_val_prob = self.model.predict(X_valid)
            val_score_macro = f1_score(y_valid, y_val_pred, average='macro')
            val_score_micro = f1_score(y_valid, y_val_pred, average='micro')
            val_p_ma = precision_score(y_valid, y_val_pred, average='macro')
            val_p_mi = precision_score(y_valid, y_val_pred, average='micro')
            val_r_ma = recall_score(y_valid, y_val_pred, average='macro')
            val_r_mi = recall_score(y_valid, y_val_pred, average='micro')
            val_auc_mi = roc_auc_score(y_valid, y_val_prob, average='micro')
            val_auc_ma = roc_auc_score(y_valid, y_val_prob, average='macro')
            # logs['F1_score_train'] = np.round(score, 5)
            logs['F1_score_val_macro'] = np.round(val_score_macro, 5)
            logs['F1_score_val_micro'] = np.round(val_score_micro, 5)
            logs['Precision_score_val_macro'] = np.round(val_p_ma, 5)
            logs['Precision_score_val_micro'] = np.round(val_p_mi, 5)
            logs['Recall_score_val_macro'] = np.round(val_r_ma, 5)
            logs['Recall_score_val_micro'] = np.round(val_r_mi, 5)
            logs['AUC_val_micro'] = np.round(val_auc_mi, 5)
            logs['AUC_val_macro'] = np.round(val_auc_ma, 5)
        else:
            logs['F1_score_train'] = np.round(score, 5)