from Simulator import Simulation, Vehicle, Client
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_data(TABLE):
    # generate episode
    Event = Simulation(TABLE)

    # initialize variables
    client = 0
    i = 0

    # store images (state and cliet requests)
    store_data = {}

    # generate first request
    client, state = Event.get_next_customer()


    while client != -1:
        # process client and store current state
        print('\rClient {}\tTime: {:.2f}'.format(i, Event.time), end="")


        # assign vehicle according to Claudia's policy
        next_vehicle = Event.rides[Event.rides.cust_id == client.ID].vehicle_id.values[0]
        V = Event.vehicles[next_vehicle-1]
        V.assign_client(client)
        
        # separate client info and vehicles info
        c_s, v_s = np.asarray(state[0]), np.asarray(state[1:])
        store_data[i] = [c_s, np.sum(v_s, axis=0), v_s[V.ID]]
        i += 1

        # update state and get next request
        client, state = Event.get_next_customer()


    # end episode: pick and drop all the remaining clients
    for v in Event.vehicles:
        print('\rFinishing with '.format(v.ID), end="")
        while len(v.clients) != 0 or len(v.clients_to_pick) != 0:
            # set time to 1000 to let the process continue until termination
            v.update(1000)
    
    store_results(TABLE, store_data, Event)

def store_results(TABLE, store_data, Event):
    # save data
    np.save('./simulation/data_compact_{}'.format(TABLE), store_data)
    print('\nFinished without problems :) \n')
    
    # store clients data
    clients_info = []
    for c in Event.customers:
        info = {
            'ID': c.ID, 'arrival': c.arrival, 'pickup_t': c.pickup_t, 'dropoff_t': c.dropoff_t
        }
        clients_info.append(info)

    df_clients = pd.DataFrame(clients_info)
    
    df_clients.to_pickle('./simulation/clients_simu_{}.pkl'.format(TABLE))
    
    comparison = pd.merge(df_clients, Event.rides, left_on='ID', right_on='cust_id', \
             suffixes=('_simu','_sol'))

    comparison.to_pickle('./simulation/comparison_{}.pkl'.format(TABLE))
    