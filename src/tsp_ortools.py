#!/usr/bin/env python3
# Copyright 2010-2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START program]
"""Simple Travelling Salesperson Problem (TSP) between cities."""

# [START import]
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
from haversine import haversine
import csv

# [END import]

def read_csv(file_path,n):
    df = pd.read_csv(file_path,nrows=n)
    places = df['Place_Name'].unique().tolist()
    coordinates = list(zip(df['Latitude'],df['Longitude']))
    return places, coordinates

def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = haversine(coordinates[i], coordinates[j])
    return distance_matrix

# [START data_model]
def create_data_model(distance_matrix):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data
    # [END data_model]


# [START solution_printer]
def print_solution(manager, routing, solution, places):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    routeList = []
    while not routing.IsEnd(index):
        routeList.append(manager.IndexToNode(index))
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    routeList.append(manager.IndexToNode(index))
    print(routeList)
    print(plan_output)
    plan_output += f"Route distance: {route_distance}miles\n"

    # Specify the filename
    filename = 'K_warmStart_40.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
         # Optionally write a header
        writer.writerow(['seq', 'place_name'])
        #write solution into csv 
        for i in routeList[:-1]:
            # Write data into the CSV file
            writer.writerow([routeList.index(i),places[i]])
    # [END solution_printer]


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]
    data_file_path = "../data/tsp_input.csv"
    places, coordinates = read_csv(data_file_path,50)
    # print(places)
    distance_matrix = calculate_distance_matrix(coordinates)
    data = create_data_model(distance_matrix)
    # [END data]

    # Create the routing index manager.
    # [START index_manager]
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    # [END index_manager]

    # Create Routing Model.2524.82
    # [START routing_model]
    routing = pywrapcp.RoutingModel(manager)

    # [END routing_model]

    # [START transit_callback]
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # [END transit_callback]

    # Define cost of each arc.
    # [START arc_cost]
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index) # Objective
    # [END arc_cost]

    # Setting first solution heuristic.
    # [START parameters]
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # search_parameters.time_limit.seconds = 1
    # [END parameters]

    # Solve the problem.
    # [START solve]
    solution = routing.SolveWithParameters(search_parameters)
    # [END solve]
    print("Solver status: ", routing.status())
    # Print solution on console.
    # [START print_solution]
    if solution:
        print_solution(manager, routing, solution, places)
    
    # [END print_solution]
   

if __name__ == "__main__":
    main()
# [END program]