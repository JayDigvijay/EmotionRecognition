# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:44:37 2020

@author: Prof. S.R.Singh
"""
import pandas as pd
from ANN import evaluate_algorithm, load_csv, dataset_minmax, normalize_dataset, back_propagation, str_column_to_int, str_column_to_float

def Fitness(gene):
    sequence = list()
    for x in gene:
        sequence.append(int(x))
    Data = dict(pd.read_csv('Emotion_Action_Units.csv'))
    Data_List = list(pd.read_csv('Emotion_Action_Units.csv'))
    Data_New = dict()
    for i in range(len(gene)):
        if(int(gene[i])):
            Data_New[Data_List[i]] = Data[Data_List[i]]
    
    Data_New[Data_List[6]] = Data[Data_List[6]]  
    Features = pd.DataFrame(Data_New)
    Features.to_csv('Selected_Features.csv', index = False)
    dataset = load_csv('Selected_Features.csv')
    for i in range(len(dataset[0])-1):
    	str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    
    n_folds = 3
    l_rate = 0.3
    n_epoch = 200
    n_hidden = 5
    
    scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

if __name__ == '__main__':
    Fitness('101101', 3, 0.3, 200, 5)