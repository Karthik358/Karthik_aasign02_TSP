import pandas as pd 
from haversine import haversine
from pulp import *
import sys
import heapq
# import pulp

import timeit
startTime = timeit.default_timer()
def read_csv(file_path,n):
    df = pd.read_csv(file_path,nrows=n)
    places = df['Place_Name'].unique().tolist()
    coordinates = list(zip(df['Latitude'],df['Longitude']))
    return places, coordinates

useWarmStart = True
if useWarmStart:
    warmStart_df = pd.read_csv('../data/warmStart/Kharagpur/K_warmStart_100.csv')

def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = haversine(coordinates[i], coordinates[j])
    return distance_matrix

def build_model(places, distance_matrix):
    # instantiate the problem - python pulp model
    # prob = pulp.LpProblem("TSP",pulp.LpMinimize)
    prob = LpProblem("TSP", LpMinimize)
    # for i in places:
    #     for j in places:
    #         if distance_matrix[places.index(i)][places.index(j)] <= 
    #Preprocess to reduce xij variables
    # List to store the results
    reduced_distance_matrix = []

    # Iterate through each list in the data
    for lst in distance_matrix:
        # Initialize a list with None values
        reduced_list = [None] * len(lst)
        
        # Find the 40 smallest values and their indices
        smallest_values_with_indices = heapq.nsmallest(80, enumerate(lst), key=lambda x: x[1])
        
        # Populate the reduced_list with the smallest values at their original indices
        for index, val in smallest_values_with_indices:
            reduced_list[index] = val
        
        # Append the reduced_list to results
        reduced_distance_matrix.append(reduced_list)

    # ****************************************
    # Defining decision variables
    # ****************************************
    x = {} # Binary x_i,j := 1 if I am visiting city j after city i; otherwise 0
    for i in places:
        for j in places:
            if i != j and reduced_distance_matrix[places.index(i)][places.index(j)] != None:
                x[(i,j)] = LpVariable('x_'+str(places.index(i)) + '_' + str(places.index(j)), cat = 'Binary')
    
    print('Xij variables',len(x))
    # print(x)
    # sys.exit()


    s = {} # Integer: s_i is the sequence number when we are visiting city i 
    # n= len(places)
    for i in places[1:]:
        s[i] = LpVariable('s_' + str(places.index(i)), cat ='Integer', lowBound=1, upBound = len(places)-1)

    if useWarmStart:
        for i in warmStart_df.index[1:]:
            seqNumber = warmStart_df['seq'][i]
            cityName = warmStart_df['place_name'][i]
            s[cityName].setInitialValue(seqNumber)
        # for i in warmStart_df.index:
        #     if i<(len(warmStart_df.index)-1) and (warmStart_df['place_name'][i],warmStart_df['place_name'][i+1]) not in x.keys():
        #         print(i,i+1)
        
       
    # ********************************************
    # Objective
    # ********************************************
    # Minimize total travel distance
    obj_val = 0
    # for i in places:
    #     for j in places:
    #         if i != j and distance_matrix[places.index(i)][places.index(j)]:
    #             obj_val += x[(i,j)]*distance_matrix[places.index(i)][places.index(j)]
    for key in x.keys():
        obj_val += x[key]*distance_matrix[places.index(key[0])][places.index(key[1])]
    prob += obj_val

    # Constraint 1
    # leaves the city exactly once
    for i in places:
        aux_sum = 0
        for j in places:
            if i != j and (i,j) in x.keys():
                aux_sum += x[(i,j)]
        prob += aux_sum ==1, 'Outgoing_sum' + str(places.index(i))

    # Constraint 2
    # enters the city exactly once
    for j in places:
        aux_sum = 0
        for i in places:
            if i != j and (i,j) in x.keys():
                aux_sum += x[(i,j)]
        prob += aux_sum ==1, 'Incoming_sum' + str(places.index(j))

    # Subtour elimination constraint
    for i in places[1:]:
        for j in places[1:]:
            if i != j and (i,j) in x.keys():
                prob += s[j] >= s[i] + 1-len(places) + len(places)*x[((i,j))],'sub_tour_'+ str(places.index(i)) + '_' + str(places.index(j))
    # *********************************
    # Solve the problem
    # ************x[(i,j)] = LpVariable('x_'+str(places.index(i)) + '_' + str(places.index(j)), cat = 'Binary')*********************
    
    # Parameter Tuning
    
    solver = 'GUROBI' 
    print('-'*50)
    print('Optimization solver', solver , 'called')
    prob.writeLP("../output/tsp.lp")
    # print(prob)
    # print()
    if solver == 'GUROBI':
        # prob.solve(GUROBI())
        prob.solve(GUROBI(warmStart=True))
    else:
        prob.solve()

    # Print the status of the solved LP
    print("Status:", LpStatus[prob.status])

    # Print the value of the variables at the optimum
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)
    sList = [places[0] for i in range(len(places)+1)]
    for i in places[1:]:
        print(s[i].varValue)
        sList[int(s[i].varValue)] = i
        # sList.insert(int(s[i].varValue),i)
    print(sList)
    88.27
    # Print the value of the objective
    print("objective=", value(prob.objective))


if __name__ == "__main__":
    data_file_path = "../data/tsp_input.csv"
    places, coordinates = read_csv(data_file_path,100)
    # print(places)
    distance_matrix = calculate_distance_matrix(coordinates)
    build_model(places, distance_matrix)

    endTime = timeit.default_timer()
 
    print(endTime - startTime)

