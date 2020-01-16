import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import skimage as sk
from skimage import measure
import math
import geopandas as gpd

import torch

import random

def generate_data(table_number):
    df_rides = pd.read_pickle('./tables/rides_sol_{}.pkl'.format(table_number))
    df_events = pd.read_pickle('./tables/events_sol_{}.pkl'.format(table_number))
    df_time = pd.read_pickle('./tables/time_sol_{}.pkl'.format(table_number))

    return df_rides, df_events, df_time

def generate_full_info(df_rides, df_time, df_events):

    # dataset with pickup and dropoff with customer and vehicle
    df_arrivals = pd.merge(df_rides, df_time, left_on='cust_id', right_on='customer')

    # filter by pickups
    df_pickups = df_events[df_events.event_id == 3]

    # add note information
    df_full_info = pd.merge(df_arrivals, df_pickups, left_on='pickup_t', right_on='time')

    df_full_info['time_x'] = df_full_info['time_x'].apply(lambda x: float(x))
    df_full_info['customer'] = df_full_info['customer'].apply(lambda x: int(x))
    df_full_info.drop(['vehicle_id_y', 'time_y', 'cust_id', 'rech_time', 'event_description', 'event_id'], axis=1, inplace=True)
    df_full_info['max_time'] = df_full_info.dropoff_t + df_full_info.excess_t

    # already has dropped the clients which were rejected by the system
    return df_full_info

def generate_graph():
    df_edges = pd.read_pickle('./tables/edges.pkl')
    df_nodes = pd.read_pickle('./tables/nodes.pkl')
    # create graph
    SF_graph = nx.convert_matrix.from_pandas_edgelist(df_edges, 'Start_node_ID', 'End_node_ID', ['distance'])

    # add position to the nodes
    for node in SF_graph.nodes:
        SF_graph.nodes[node]['lat'] = df_nodes[df_nodes.ID == node].rel_lat.values[0]
        SF_graph.nodes[node]['lon'] = df_nodes[df_nodes.ID == node].rel_lon.values[0]

    return SF_graph

def generate_SF_graph():
    df_edges = pd.read_pickle('./tables/edges.pkl')
    df_nodes = pd.read_pickle('./tables/nodes.pkl')

    mean_lat = sum(df_nodes.Lat) / len(df_nodes)
    mean_lon = sum(df_nodes.Lon) / len(df_nodes)

    dist = 1500
    transport_mode = "drive"
    SF = ox.graph_from_point((mean_lat, mean_lon), distance=dist, distance_type='bbox', network_type=transport_mode)

    travel_speed = 10

    # add an edge attribute for time in minutes required to traverse each edge
    meters_per_minute = travel_speed * 1000 / 60  # km/h to m/min
    for u, v, k, data in SF.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute

    return SF


def map_nodes(G, df):
    """
    Map nodes to the closest node in G
    :param G: osmnx graph
    :param df: pandas dataframe with nodes information
    :return: dictionary node from df: closest node in G
    """
    mapping = {}
    inv_mapping = {}
    for i in range(len(df)):
        n_ref_org = df.iloc[i].ID
        n_ref_osmnx = ox.get_nearest_node(G, point=(df.iloc[i].Lat, df.iloc[i].Lon))
        mapping[n_ref_org] = n_ref_osmnx
        inv_mapping[n_ref_osmnx] = n_ref_org


    return mapping, inv_mapping



def get_vehicle_assignment_nn(state, model, num_v=30, im_size=30, channels_net=31, single_output=False, single_input=False):
    """
    Generate vehicle assignment based on current information

    :param client: vector [current location, future location]
    :param state: vector size=[num_vehicles; 6]; v_i=[load, queue, current location, future location]
    :return: integer; vehicle number assigned to pickup client
    """

    min_lat, max_lat = 37.772949, 37.790377
    d_lat = max_lat - min_lat
    min_lon, max_lon = -122.424296, -122.405883
    d_lon = max_lon - min_lon

    nv = num_v
    dist = 1 / (im_size - 1)

    x = np.zeros((1, channels_net, im_size, im_size))
    x_aux = np.zeros((1, nv, 6))

    x[0][0][int((state[0][1] - min_lat)/d_lat // dist)][int((state[0][0] - min_lon)/d_lon // dist)] = 1
    x[0][0][int((state[0][3] - min_lat)/d_lat // dist)][int((state[0][2] - min_lon)/d_lon // dist)] = -1

    #x_aux += [(state[0][1] - min_lat)/d_lat, (state[0][0] - min_lon)/d_lon,
    #            (state[0][3] - min_lat)/d_lat , (state[0][2] - min_lon)/d_lon]

    if channels_net == 31:
        
        for i in range(nv):
            try:
                x[0][i+1][int((state[1+i][3] - min_lat)/d_lat // dist)][int((state[1+i][2] - min_lon)/d_lon // dist)] = 1
                x_aux[0][i] = [state[1+i][0], state[1+i][1], (state[1+i][3] - min_lat)/d_lat, (state[1+i][2] - min_lon)/d_lon,
                          (state[1+i][5] - min_lat)/d_lat , (state[1+i][4] - min_lon)/d_lon]
            except:
                x[0][i+1][int(0.89818683 // dist)][int(0.58334329 // dist)] = 1
                x_aux[0][i] = [0, 0, 0.89818683, 0.58334329, 0.89818683, 0.58334329]
                #x_aux += [0, 0, 0.82, 0.69, 0.82, 0.69]
    elif channels_net == 61:
        for i in range(1,nv+1):
            try:
                x[0][2*i-1][int((state[1+i][3] - min_lat)/d_lat // dist)][int((state[1+i][2] - min_lon)/d_lon // dist)] = 1
                x[0][2*i][int((state[1+i][5] - min_lat)/d_lat // dist)][int((state[1+i][4] - min_lon)/d_lon // dist)] = -1
                x_aux[0][i-1] = [state[1+i][0], state[1+i][1], (state[1+i][3] - min_lat)/d_lat, (state[1+i][2] - min_lon)/d_lon,
                          (state[1+i][5] - min_lat)/d_lat , (state[1+i][4] - min_lon)/d_lon]
            except:
                x[0][2*i-1][int(0.89818683 // dist)][int(0.58334329 // dist)] = 1
                x[0][2*i][int(0.89818683 // dist)][int(0.58334329 // dist)] = -1
                x_aux[0][i-1] = [0, 0, 0.89818683, 0.58334329, 0.89818683, 0.58334329]
    elif channels_net == 2:
        for i in range(nv):
            try:
                x[0][1][int((state[1+i][3] - min_lat)/d_lat // dist)][int((state[1+i][2] - min_lon)/d_lon // dist)] = 1
                x_aux[0][i] = [state[1+i][0], state[1+i][1], (state[1+i][3] - min_lat)/d_lat, (state[1+i][2] - min_lon)/d_lon,
                          (state[1+i][5] - min_lat)/d_lat , (state[1+i][4] - min_lon)/d_lon]
            except:
                x[0][1][int(0.89818683 // dist)][int(0.58334329 // dist)] = 1
                x_aux[0][i] = [0, 0, 0.89818683, 0.58334329, 0.89818683, 0.58334329]
    else:
        print('Invalid number of channels')
        
              


    x_aux = np.asarray(x_aux)
    #x_aux.flatten()

    x_aux = torch.tensor(x_aux).type(torch.FloatTensor)
    x = torch.tensor(x).type(torch.FloatTensor)
    if single_output:
        if single_input:
            output2 = model(x)
        else:
            output2 = model(x, x_aux)
    else:
        if single_input:
            output1, output2 = model(x)
        else:
            output1, output2 = model(x, x_aux)

    _, a = torch.max(output2, 1)

    v_characteristics = x_aux[0][a.item()]
    possible_c = np.where(x_aux[0] == v_characteristics)
    candidates = [i for i in set(possible_c[0]) if len(np.where(possible_c[0] == i)[0]) == 6]
    #random.shuffle(candidates)
    return a
    #return candidates[0]



def get_vehicle_assignment_random(num_v=30):
    return np.random.randint(num_v)
























