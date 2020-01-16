import numpy as np
import pandas as pd


class Arrival: 
    
    def __init__(self, df_time, df_events, df_rides):
        """
        df_time: pandas df with customer and request arrival time
        df_events: pandas df with all the events in a simulation
        df_rides: pandas df with info per vehicle
        """
        df_arrivals = pd.merge(df_rides, df_time, left_on='cust_id', right_on='customer')
        df_arrivals['max_time'] = df_arrivals['excess_t'] + df_arrivals['dropoff_t'] 

        self.pickups = df_events[df_events.event_id == 3]
        self.rides = df_rides
        self.arrivals = df_arrivals
        
        self.customer = 0
        self.max_customer = int(df_time.customer.max())
        
    def get_new_arrival(self):
        
        if self.customer < self.max_customer:
            self.customer += 1
        
        cust_info = self.arrivals[self.arrivals.customer == str(self.customer)]
        
        request_time = float(cust_info.time.values[0])
        max_time = float(cust_info.max_time.values[0])
        
        
        pickup_time = cust_info.pickup_t.values[0]
    
        node = self.pickups[self.pickups.time == pickup_time].node.values[0]
        
        return [node, request_time, max_time]
        
        