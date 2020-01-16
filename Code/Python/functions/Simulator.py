import queue

from Utils import *

class Client:

    def __init__(self, origin, arrival_time, mapping, g):
        self.ID = origin
        self.origin = origin
        self.destination = origin + 100
        self.arrival = arrival_time
        self.pickup_t = 1000
        self.dropoff_t = 1000
        self.G = g

        self.ar_node = mapping[self.origin]
        self.dest_node = mapping[self.destination]


    def __str__(self):
        return 'Client {} from {} to {}. Arrival {}'.format(self.ID, self.origin, self.destination, self.arrival)


    def get_representation(self):
        current_n = self.G.nodes[self.ar_node]
        dest_n = self.G.nodes[self.dest_node]
        return [current_n['x'], current_n['y'], dest_n['x'], dest_n['y']]
        #im = ImageRepresentation()
        #return im.client(self)



class ImageRepresentation:
    def __init__(self):
        SF_graph = generate_graph()
        self.max_lat = max(SF_graph.nodes[i]['lat'] for i in SF_graph.nodes)
        self.max_lon = max(SF_graph.nodes[i]['lon'] for i in SF_graph.nodes)

        self.size = max(self.max_lon, self.max_lat)
        self.matrix = np.zeros((int(self.size + 1), int(self.size + 1)))

        self.graph = SF_graph

    def small_representation(self, matrix=np.array([])):
        if matrix.shape[0] == 0:
            matrix = self.matrix
        return measure.block_reduce(matrix, (3,3), np.max)

    def client(self, client):
        m = self.matrix.copy()
        p_lat = self.graph.nodes[client.ID]['lat']
        p_lon = self.graph.nodes[client.ID]['lon']
        m[int(p_lat), int(p_lon)] = 1 # add a 1 for the pickup location
        d_lat = self.graph.nodes[client.ID + 100]['lat']
        d_lon = self.graph.nodes[client.ID + 100]['lon']
        m[int(d_lat), int(d_lon)] = -1  # add a -1 for the dropoff location
        return self.small_representation(m)

    def vehicle(self, vehicle):
        m = self.matrix.copy()
        lat = self.graph.nodes[vehicle.current_node]['lat']
        lon = self.graph.nodes[vehicle.current_node]['lon']
        #add a 1 for the current position
        m[int(lat), int(lon)] = 1
        return self.small_representation(m)



class Vehicle:

    def __init__(self, ID, SF_graph, mapping, inv_mapping):
        self.ID = ID
        self.current_node = 221 # change; where are all the vehicles initially located?
        self.destination = 221
        self.plan = []
        self.plan_time = []
        self.clients = {} #key: clientID, value: client_destination
        self.clients_to_pick = {}
        #self.events = queue.Queue() #queue of events (pickup and dropoff) to handle
        self.G = SF_graph
        self.time = 0
        self.mapping = mapping
        self.inv_mapping = inv_mapping

    def __str__(self):
        return 'Vehicle {} with destination {} and {} clients and {} future clients'.format(self.ID, self.destination, len(self.clients), len(self.clients_to_pick))

    def next_destination(self):
        if not self.clients and not self.clients_to_pick:
            self.destination = 221 # or any recharging station (the closest one)
        else:
            distance = math.inf
            next_destination = self.current_node
            for client in self.clients:
                #compute distance to client destination
                distance_client = nx.shortest_path_length(self.G, self.mapping[self.current_node], self.mapping[self.clients[client]], 'distance')
                if distance_client <= distance:
                    distance = distance_client
                    next_destination = self.clients[client]
            for client in self.clients_to_pick:
                # compute distance to client destination
                distance_client = nx.shortest_path_length(self.G, self.mapping[self.current_node], self.mapping[self.clients_to_pick[client]], 'distance')
                if distance_client <= distance:
                    distance = distance_client
                    next_destination = self.clients_to_pick[client]


            self.destination = next_destination

        # update plan

        # compute shortest path
        plan_osmnx = nx.shortest_path(self.G, self.mapping[self.current_node], self.mapping[self.destination], 'distance')
        self.plan_time = [0]  # restart plan time
        self.plan = []  # restart plan

        #save nodes to visit in original syst of ref
        for n in plan_osmnx:
            if n in self.inv_mapping:
                self.plan.append(self.inv_mapping[n])

        # ensure destination node when multiple nodes are possible
        self.plan[-1] = self.destination

        # update time for plan
        t = 0
        for i in range(len(self.plan) -1):
            t = nx.shortest_path_length(self.G, self.mapping[self.plan[i]], self.mapping[self.plan[i +1]], weight='time')
            self.plan_time.append( self.plan_time[-1] + t)

    def assign_client(self, client):
        """
        Add client to the vehicle list of future pickups
        :param client: Client to be picked up
        """
        # add the client to the list of clients to pick
        self.clients_to_pick[client] = client.origin
        self.next_destination()


    def pickup_client(self, client):
        """
        Add client to the vehicle and update position
        :param client: Client to be picked up
        """
        #print('Picking client')
        del self.clients_to_pick[client] # delete from future pickups
        #client.pickup_t = self.time
        self.clients[client] = client.destination # add to future dropoffs
        self.current_node = client.origin
        client.pickup_t = self.time

        # update plan
        self.next_destination()
        #TODO: when calling new_client() we also have to call next_destination; otherwise it won't update the path


    def drop_client(self, client):
        """
        Remove client from the vehicle and update position
        :param client: Client to be dropped
        """
        #print('Dropping client')
        #self.current_node = client.destination
        #client.dropoff_t = self.time
        del self.clients[client]
        client.dropoff_t = self.time
        # update plan
        self.next_destination()

    def update(self, t):
        """
        Update vehicles' position
        :param t: time
        :return:
        """
        delta_t = t - self.time
        #check if there is a plan to be completed
        if len(self.plan) == 0:
            self.time = t

        else:
            #while we can complete the plan time
            while self.time + self.plan_time[-1] < t:
                #case 1: we can complete the full plan in the given time step
                #update location
                self.current_node = self.destination
                #update time
                self.time += self.plan_time[-1]
                # dropoff clients
                current_c = list(self.clients)
                for client in current_c:
                    if self.clients[client] == self.current_node:
                        self.drop_client(client)
                #pickup clients
                future_c = list(self.clients_to_pick)
                for client in future_c:
                    if client.origin == self.current_node:
                        self.pickup_client(client)
                # new plan has been recomputed; complete if possible

            # we cannot complete the current plan but still can move
            delta_t = t - self.time
            # get reachable cities
            time_to_cities = list(filter(lambda x: x < delta_t, self.plan_time))
            # move to furthest node
            if len(time_to_cities) > 0:
                next_city = self.plan[self.plan_time.index(time_to_cities[-1])]
                # update time and current location
                self.time += time_to_cities[-1]
                self.current_node = next_city

            #update plan
            self.next_destination()



    def get_representation(self):
        st_node = self.G.nodes[self.mapping[self.current_node]]
        x_current, y_current = st_node['x'], st_node['y']
        
        next_node = self.G.nodes[self.mapping[self.destination]]
        x_next, y_next = next_node['x'], next_node['y']
        
        capacity = 5 - len(self.clients)
        future_c = len(self.clients_to_pick)
        return [x_current, y_current, x_next, y_next, capacity, future_c]
        
        #im = ImageRepresentation()
        #return im.vehicle(self)




class Simulation:

    def __init__(self, table_number):
        self.time = 0

        #information storage
        df_rides, df_events, df_time = generate_data(table_number)
        self.pickups = df_events[df_events.event_id == 3]
        self.dropoff = df_events[df_events.event_id == 4]
        self.rides = df_rides
        self.full_info = generate_full_info(df_rides, df_time, df_events)

        # graphs and nodes
        self.graph = generate_graph()
        self.G = generate_SF_graph()
        self.nodes_df = pd.read_pickle('./tables/nodes.pkl')

        # mapping: map every node in Claudia's list to a node in the graph
        self.mapping, self.inv_mapping = map_nodes(self.G, self.nodes_df)

        #customers arrival
        self.max_customer = int(df_time.customer.max())
        self.customers = []
        self.current_customer = 0
        self.initiate_customers()


        #vehicles
        self.num_vehicles = max(self.full_info.vehicle_id_x)
        self.vehicles = []
        self.initiate_vehicles()



    def initiate_vehicles(self):
        """
        initializes the list of vehicles based on the number of vehicles given by the tables
        """
        for i in range(self.num_vehicles):
            #add vehicle Objects to the list of vehicles
            self.vehicles.append(Vehicle(i, self.G, self.mapping, self.inv_mapping))

    def initiate_customers(self):
        """
        initializes the list of customers based on the information of the tables
        """
        for c in self.full_info.customer:
            client_info = self.full_info[self.full_info.customer == c]
            client_instance = Client(c, client_info.time_x.values[0], self.mapping, self.G) #i=ID and pickup node
            self.customers.append(client_instance)

    def get_next_customer(self):
        """
        updates current customer and time information
        """
        if self.current_customer >= len(self.customers) - 1:
            return -1, -1
        else:
            representation = []
            self.current_customer += 1
            client = self.customers[self.current_customer]
            representation.append(client.get_representation())
            self.time = client.arrival

            #update vehicles position
            for v in self.vehicles:
                v.update(self.time)
                representation.append(v.get_representation())

            return (client, representation)



class Scenario:

    def __init__(self, table_number):
        # instantiate simulation
        Event = Simulation(table_number)
        client = 0
        while client != -1:
            #get next customer
            client, state = Event.get_next_customer()
            #TODO: call NN to assign vehicle
            next_vehicle = Event.rides[Event.rides.cust_id == client.ID].vehicle_id
            print(next_vehicle)
            V = Event.vehicles[next_vehicle-1]
            V.assign_client(client)










