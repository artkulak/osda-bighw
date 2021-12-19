from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax
import numpy as np
import pandas as pd

class LazyLearningAlgo:
    '''
    Set of Lazy Learning Algorithm based on FCA frequent itemsets concept
    '''
    
    def __init__(self, min_support, allow_online_learning = False, fca_algorithm = 'fpgrowth', prediction_algorithm = 'most_itemsets'):
        '''
            :param min_support: min support value for closed itemsets
            :param allow_online_learning: if you want to include rows of the test set to predict the next rows of the test set (kinda pseudo labelling)
            :param fca_algorithm: type of closed items mining algorithm (available: fpgrowth, fpmax, fpcommon, apriori)
            :param prediction_algorithm: type of prediction algorithm (available: most_itemsets, itemsets_intersection_sums, itemsets_intersection_probas)
        '''
        self.min_support = min_support
        self.allow_online_learning = allow_online_learning
        
        assert(fca_algorithm in ['fpgrowth', 'fpmax', 'apriori'])
        self.fca_algorithm = fca_algorithm
        
        assert(prediction_algorithm in ['most_itemsets', 'itemsets_intersection_sums', 'itemsets_intersection_probas'])
        self.prediction_algorithm = prediction_algorithm
        
    
    def _find_frequent_itemsets(self, data):
        '''
        Find closed itemsets with fpgrowth algorithm
        :param data: binarized data where most frequent closed itemsets will be found
        '''
        
        if self.fca_algorithm == 'fpgrowth':
            return fpgrowth(data, min_support=self.min_support, use_colnames=True)
        elif self.fca_algorithm == 'fpmax':
            return fpmax(data, min_support=self.min_support, use_colnames=True)
        elif self.fca_algorithm == 'apriori':
            return apriori(data, min_support=self.min_support, use_colnames=True)
        
        return fpgrowth(data, min_support=self.min_support, use_colnames=True)
            
            
        
    
    def _algo_most_itemsets(self, X_train, Y_train, X_test):
        '''
            Classify each example from test set based on how many itemsets match frequent_zero_class itemsets and match frequent_one_class itemsets
            Think of, |𝑔+∩𝑔| vs |𝑔-∩𝑔|
            :param X_train: binarized items of the train set
            :param Y_train: target values for X_train
            :param X_test: binarized items of the test set
        '''
        
        frequent_zero_class = self._find_frequent_itemsets(X_train[Y_train == 0])
        frequent_one_class = self._find_frequent_itemsets(X_train[Y_train == 1])
        
        test_classes = []
        for (i, row) in X_test.iterrows():
            total_matches_one = 0
            for item in frequent_one_class['itemsets']:
                columns = np.array(list(item))
                if X_test.loc[i][columns].sum() == len(columns):
                    total_matches_one += 1

            total_matches_zero = 0
            for item in frequent_zero_class['itemsets']:
                columns = np.array(list(item))
                if X_test.loc[i][columns].sum() == len(columns):
                    total_matches_zero += 1

            if total_matches_one > total_matches_zero:
                output_label = 1
            elif total_matches_one < total_matches_zero:
                output_label = 0
            else:
                output_label = 1
            test_classes.append(output_label)
            
            if self.allow_online_learning:
                X_train = X_train.append(X_test.iloc[i:i+1, :]).reset_index(drop = True)
                Y_train = Y_train.append(pd.Series([output_label])).reset_index(drop = True)
                
                frequent_zero_class = self._find_frequent_itemsets(X_train[Y_train == 0])
                frequent_one_class = self._find_frequent_itemsets(X_train[Y_train == 1])
                
        return np.array(test_classes)
    
    def _algo_itemsets_intersection_sums(self, X_train, Y_train, X_test):
        '''
            Classify each example from test set based on the total sums of proportion of matching each itemset
            to the items found for the validation sample
            :param X_train: binarized items of the train set
            :param Y_train: target values for X_train
            :param X_test: binarized items of the test set
        '''
        
        frequent_zero_class = self._find_frequent_itemsets(X_train[Y_train == 0])
        frequent_one_class = self._find_frequent_itemsets(X_train[Y_train == 1])
        
        test_classes = []
        for (i, row) in X_test.iterrows():
            total_matches_one = 0
            for item in frequent_one_class['itemsets']:
                columns = np.array(list(item))
                total_matches_one += X_test.loc[i][columns].sum() / len(columns)

            total_matches_zero = 0
            for item in frequent_zero_class['itemsets']:
                columns = np.array(list(item))
                total_matches_zero += X_test.loc[i][columns].sum() / len(columns)

            if total_matches_one > total_matches_zero:
                output_label = 1
            elif total_matches_one < total_matches_zero:
                output_label = 0
            else:
                output_label = 1
            test_classes.append(output_label)
            
            if self.allow_online_learning:
                X_train = X_train.append(X_test.iloc[i:i+1, :]).reset_index(drop = True)
                Y_train = Y_train.append(pd.Series([output_label])).reset_index(drop = True)
                
                frequent_zero_class = self._find_frequent_itemsets(X_train[Y_train == 0])
                frequent_one_class = self._find_frequent_itemsets(X_train[Y_train == 1])
                
        return np.array(test_classes)
    
    def _algo_itemsets_intersection_probas(self, X_train, Y_train, X_test):
        '''
            Classify each example from test set based on the multiple of probabilities of itemset match the prediction item 
            :param X_train: binarized items of the train set
            :param Y_train: target values for X_train
            :param X_test: binarized items of the test set
        '''
        
        frequent_zero_class = self._find_frequent_itemsets(X_train[Y_train == 0])
        frequent_one_class = self._find_frequent_itemsets(X_train[Y_train == 1])
        
        test_classes = []
        for (i, row) in X_test.iterrows():
            total_matches_one = 1
            for item in frequent_one_class['itemsets']:
                columns = np.array(list(item))
                total_matches_one *= X_test.loc[i][columns].sum() / len(columns)

            total_matches_zero = 1
            for item in frequent_zero_class['itemsets']:
                columns = np.array(list(item))
                total_matches_zero *= X_test.loc[i][columns].sum() / len(columns)

            if total_matches_one > total_matches_zero:
                output_label = 1
            elif total_matches_one < total_matches_zero:
                output_label = 0
            else:
                output_label = 1
            test_classes.append(output_label)
            
            if self.allow_online_learning:
                X_train = X_train.append(X_test.iloc[i:i+1, :]).reset_index(drop = True)
                Y_train = Y_train.append(pd.Series([output_label])).reset_index(drop = True)
                
                frequent_zero_class = self._find_frequent_itemsets(X_train[Y_train == 0])
                frequent_one_class = self._find_frequent_itemsets(X_train[Y_train == 1])
                
        return np.array(test_classes)
    
    def run_prediction(self, X_train, Y_train, X_test):
        '''
            Classify each example from test set using the selected algorithm
            :param X_train: binarized items of the train set
            :param Y_train: target values for X_train
            :param X_test: binarized items of the test set
        '''
        if self.prediction_algorithm == 'most_itemsets':
            return self._algo_most_itemsets(X_train, Y_train, X_test)
        elif self.prediction_algorithm == 'itemsets_intersection_sums':
            return self._algo_itemsets_intersection_sums(X_train, Y_train, X_test)
        elif self.prediction_algorithm == 'itemsets_intersection_probas':
            return self._algo_itemsets_intersection_probas(X_train, Y_train, X_test)
        
        return self._algo_most_itemsets(X_train, Y_train, X_test)