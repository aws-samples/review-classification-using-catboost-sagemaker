import numpy as np
import os
import matplotlib
import pandas as pd

matplotlib.use('agg', warn=False, force=True)
from matplotlib import pyplot as plt


class MetricsCalculator(object):

    def __init__(self):
        script_dir = os.path.dirname(__file__)
        self.images_dir = os.path.join(script_dir, 'images/')
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)

    #### Threshold Analysis

    def threshold_cert(self, labels, predictions):
        result_cert = []
        for i in [x / 100.0 for x in range(0, 110, 5)]:
            predicted_0 = np.where(predictions <= i, 1, 0).sum()

            actual_0 = np.where(labels == 0, 1, 0).sum()
            actual_1 = np.where(labels == 1, 1, 0).sum()

            pred_0_actual_0 = np.where((predictions <= i) & (labels == 0), 1, 0).sum()

            pred_0_actual_1 = np.where((predictions <= i) & (labels == 1), 1, 0).sum()

            Recall = pred_0_actual_0 * 100 / actual_0
            FDR = (1 - pred_0_actual_0 / predicted_0) * 100
            FPR = pred_0_actual_1 * 100 / actual_1

            res = [i] + [Recall, FDR, FPR]
            # results = analyze_volume(i)
            result_cert.append(res)
            # print (i , res)
        result_cert = pd.DataFrame(result_cert)
        result_cert.columns = ['Threshold', 'True Possitive rate', 'False Discovery Rate', 'False Positive rate']
        return result_cert

    def threshold_decert(self, labels, predictions):
        result_decert = []
        for i in [x / 100.0 for x in range(0, 101, 5)]:
            predicted_1 = np.where(predictions > i, 1, 0).sum()

            actual_1 = np.where(labels == 1, 1, 0).sum()
            actual_0 = np.where(labels == 0, 1, 0).sum()

            pred_1_actual_1 = np.where((predictions > i) & (labels == 1), 1, 0).sum()
            pred_1_actual_0 = np.where((predictions > i) & (labels == 0), 1, 0).sum()

            Recall = pred_1_actual_1 * 100 / actual_1
            FDR = (1 - pred_1_actual_1 / predicted_1) * 100
            FPR = (pred_1_actual_0 / actual_0) * 100

            res = [i] + [Recall, FDR, FPR]
            # results = analyze_volume(i)
            result_decert.append(res)
            # print (i , res)
        result_decert = pd.DataFrame(result_decert)
        result_decert.columns = ['Threshold', 'True Possitive Rate', 'False Discovery rate', 'False Possitve rate']
        return result_decert

    def run_metrics(self, labels, predictions):
        from sklearn import metrics

        result = list()
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        roc_auc_score = metrics.roc_auc_score(labels, predictions)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(roc_auc_score))
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.legend(loc=4)
        plt.show()
        plt.savefig(self.images_dir + 'roc_curve.png')

        prediction_res = list()

        print("ROC AUC Score : ", roc_auc_score)

        prediction_res.append(roc_auc_score)
        prediction_res.append(roc_auc_score)
        LB = [x / 100.0 for x in range(0, 101, 5)]
        UB = [x / 100.0 for x in range(0, 110, 5)]

        predict_flip = predictions
        target_flip = 1-labels
        #print(LB)
        #print(UB)
        pass_fc = []
        for i in range(0,len(LB)):
            #print(LB[i], UB[i+1])
            Total_count = np.where((predict_flip.astype(float) >= LB[i]) &  (predict_flip.astype(float) < UB[i+1]), 1,0 ).sum()
            Total_pass = np.where((predict_flip.astype(float) >= LB[i]) &  (predict_flip.astype(float) < UB[i+1]), target_flip,0 ).sum()
            pass_fc.append([LB[i] ,UB[i+1],Total_count , Total_pass , np.round(Total_pass/Total_count,4)])
            #print(Total_count , Total_pass , Total_pass/Total_count)

        pass_fc = pd.DataFrame(pass_fc)
        pass_fc.columns = ['LB', 'UB','Count','Pass','Pass_rate']


        precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
        auc = metrics.auc(recall, precision)
        print("PR-AUC : ", auc)

        decert = str(self.threshold_decert(labels, predictions).to_dict())
        cert = str(self.threshold_cert(labels, predictions).to_dict())

        predictions = np.round(predictions)

        confusion_matrix = metrics.confusion_matrix(labels, predictions)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion_matrix)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.images_dir + 'confusion_matrix.png')
        print(" confusion matrix: ", confusion_matrix)

        recall_score = metrics.recall_score(labels, predictions, pos_label=1)
        print("Recall Score : ", recall_score)
        prediction_res.append(recall_score)

        precison_score = metrics.precision_score(labels, predictions, pos_label=1)
        print("Precison Score : ", precison_score)
        prediction_res.append(precison_score)

        f1_score = metrics.f1_score(labels, predictions, pos_label=1)
        print("F1 Score : ", f1_score)
        prediction_res.append(f1_score)

        accuracy_score = metrics.accuracy_score(labels, predictions)
        print("Accuracy score: ", accuracy_score)
        prediction_res.append(accuracy_score)

        result.append(prediction_res)
        print('The result of prediction metrics is {}'.format(result))

        metrics = {
            'confusion_matrix': {
                'TN': str(confusion_matrix[0, 0]),
                'FP': str(confusion_matrix[0, 1]),
                'FN': str(confusion_matrix[1, 0]),
                'TP': str(confusion_matrix[1, 1])
            },
            'recall_score': str(recall_score),
            'precision_score': str(precison_score),
            'F1_Score': str(f1_score),
            'accuracy_score': str(accuracy_score),
            'auc': str(auc),
            'roc_auc_score': str(roc_auc_score),
            'threshold_decert': decert,
            'threshold_cert': cert,
            'pass_rate': str(pass_fc.to_dict())
        }

        return metrics
