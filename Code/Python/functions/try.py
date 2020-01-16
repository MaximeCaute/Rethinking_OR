from Simulator import Simulation


def main():
    Event = Simulation(4)
    client = 0

    i = 0
    client, state = Event.get_next_customer()
    while client != -1:
        print('Client ', i, 'Time: ', Event.time)

        next_vehicle = Event.rides[Event.rides.cust_id == client.ID].vehicle_id.values[0]
        V = Event.vehicles[next_vehicle - 1]
        V.assign_client(client)
        i += 1
        client, state = Event.get_next_customer()

    for v in Event.vehicles:
        print('Finishing with ', v.ID)

        while len(v.clients) != 0 or len(v.clients_to_pick) != 0:
            v.update(10000)

    print('Finished without problems :) \n')
    for V in Event.vehicles:
        print(V)




if __name__ == "__main__":
    main()