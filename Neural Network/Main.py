# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
from GA import GA
from Fitness_Function import Fitness
import pickle
from ANN import accuracy_metric, load_csv, dataset_minmax, normalize_dataset, str_column_to_int, str_column_to_float, predict
'''
Selection_Param = int(input("Please choose the selection function. Type:\n '0' for Roulette Selection \n '1' for Tournament Selection\n"))
Crossover_Param = int(input("Please choose the crossover function. Type:\n '0' for Uniform Crossover \n '1' for Single Point Crossover \n '2' for Multi-Point Crossover\n"))
Mutation_Param = int(input("Please choose the mutation function. Type:\n '0' for Inversion \n '1' for Single Point Mutation\n"))
'''
#Optimal_Feature_Subset = GA(Selection_Param, Crossover_Param, Mutation_Param)
Optimal_Feature_Subset = '101111'
Fitness(Optimal_Feature_Subset)


pickfile = open('NN_Model.pickle', 'rb')
Network = pickle.load(pickfile)
pickfile.close()

Data = dict(pd.read_csv('Test_Data.csv'))
Data_List = list(pd.read_csv('Test_Data.csv'))
Data_New = dict()
for i in range(len(Optimal_Feature_Subset)):
    if(int(Optimal_Feature_Subset[i])):
        Data_New[Data_List[i]] = Data[Data_List[i]]

Data_New[Data_List[6]] = Data[Data_List[6]]  
Features = pd.DataFrame(Data_New)
Features.to_csv('Selected_Features.csv', index = False)
test = load_csv('Selected_Features.csv')

for i in range(len(test[0])-1):
	str_column_to_float(test, i)
# convert class column to integers
str_column_to_int(test, len(test[0])-1)
# normalize input variables
minmax = dataset_minmax(test)
normalize_dataset(test, minmax)

test_set = list()
predictions = list()
for row in test:
    row_copy = list(row)
    test_set.append(row_copy)
    row_copy[-1] = None
    prediction = predict(Network, row_copy)
    predictions.append(prediction)
print(predictions)

actual = [row[-1] for row in test_set]
accuracy = accuracy_metric(actual, predictions)
print(accuracy, '%')