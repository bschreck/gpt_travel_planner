from dotenv import load_dotenv
from itertools import permutations
from datetime import timedelta, date
import os
from duffel_api import Duffel
import pandas as pd
from ortools.sat.python import cp_model
from concurrent.futures import ThreadPoolExecutor
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random_exponential,
    stop_after_attempt,
)
from duffel_api.http_client import ApiError
from requests import HTTPError
from pyflightdata import FlightData
from haversine import haversine, Unit
import networkx as nx
from heapq import heappush, heappop
from functools import cache


@cache
def get_airport_flights(airport_code, arrivals=True):
    f=FlightData()
    f.login(os.environ['FLIGHT_RADAR_EMAIL'], os.environ['FLIGHT_RADAR_PASSWORD'])
    # TODO paginate
    if arrivals:
        flights_raw = f.get_airport_arrivals(airport_code)
    else:
        flights_raw = f.get_airport_departures(airport_code)
    flights = []
    for flight in flights_raw:
        origin = flight['origin']['code']['iata']
        destination = flight['destination']['code']['iata']
        airline_code = flight['owner']['code']['iata']
        airline_name = flight['owner']['name']
        arrival_time = flight['time']['scheduled']['arrival']
        departure_time = flight['time']['scheduled']['departure']
        operator_name = flight['airline']['name']
        operator_code = flight['airline']['code']['iata']
        origin_latitude = flight['origin']['position']['latitude']
        origin_longitude = flight['origin']['position']['longitude']
        dest_latitude = flight['destination']['position']['latitude']
        dest_longitude = flight['destination']['position']['longitude']
        distance = haversine(
            (origin_latitude, origin_longitude),
            (dest_latitude, dest_longitude),
            unit=Unit.MILES
        )
        flights.append(
            {
                'Airline': airline_name,
                'Airline Code': airline_code,
                'Operator': operator_name,
                'Operator Code': operator_code,
                'Cost': 0,
                'Origin': origin,
                'Destination': destination,
                'Departure Time': departure_time,
                'Arrival Time': arrival_time,
                'Stops': 0,
                'Origin Latitude': origin_latitude,
                'Origin Longitude': origin_longitude,
                'Destination Latitude': dest_latitude,
                'Destination Longitude': dest_longitude,
                'Distance': distance,
            }
        )
        return pd.DataFrame(flights).sort_values('Distance')


def find_1_stop_flights(origin, destination, direct_flights_time):
    # Priority queue for candidate flights, sorted by total travel time
    candidate_flights = []

    # Set to keep track of visited airports to prevent re-visiting
    visited_airports = set()

    # Get initial flights from the origin
    origin_flights = get_airport_flights(origin, arrivals=False)
    for _, flight in origin_flights.iterrows():
        visited_airports.add(flight['Destination'])
        heappush(candidate_flights, (flight['Travel Time'], [flight]))

    # List to store valid 1-stop connections
    valid_connections = []

    while candidate_flights:
        total_travel_time, flights = heappop(candidate_flights)
        last_flight = flights[-1]

        # Check if the current path can be completed with a flight to the destination
        if last_flight['Destination'] != destination:
            # Get connecting flights from the last airport
            connecting_flights = get_airport_flights(last_flight['Destination', arrivals=False])
            for _, conn_flight in connecting_flights.iterrows():
                # Avoid cycles by skipping visited airports
                if conn_flight['Destination'] in visited_airports:
                    continue

                # Calculate layover time
                layover_time = max((conn_flight['Departure Time'] - last_flight['Arrival Time']).total_seconds() / 3600, 0)

                # Total travel time with layover
                new_total_time = total_travel_time + conn_flight['Travel Time'] + layover_time

                # Check the total travel time so far against the direct flight time
                if new_total_time <= direct_flights_time + 5 * 60 and layover_time >= 1:
                    # If this path is valid, add it to the list of valid connections
                    new_flights = flights + [conn_flight]
                    heappush(candidate_flights, (new_total_time, new_flights))
                    visited_airports.add(conn_flight['Destination'])

        else:
            # If we have reached the destination, calculate the total travel time
            # and check if the layover is at least 1 hour
            if len(flights) == 2:
                first_flight = flights[0]
                layover_time = (last_flight['Departure Time'] - first_flight['Arrival Time']).total_seconds() / 3600
                if layover_time >= 1:
                    # Save this connection as a valid option
                    valid_connections.append((total_travel_time, flights))

            # Stop adding new airports once we exceed the time limit
            if total_travel_time > direct_flights_time + 5 * 60:
                break

    # Sort the valid connections by total travel time
    valid_connections.sort(key=lambda x: x[0])

    return valid_connections




load_dotenv()
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v}={self.Value(v)}", end=" ")
        print()

    def solution_count(self):
        return self.__solution_count


def search_flights(origin, destination, departure_date, max_connections=1, cabin_class="economy", passengers=1, currency="USD"):
    print(f"Searching for flights from {origin} to {destination} on {departure_date}")
    # TODO partial offers
    duffel = Duffel(access_token=os.environ["DUFFEL_API_KEY"])

    @retry(
        wait=wait_random_exponential(multiplier=1, max=64),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ApiError, HTTPError)),
    )
    def reqfn():
        return (
            duffel.offer_requests.create()
            .cabin_class(cabin_class)
            .max_connections(max_connections)
            .return_offers()
            .passengers([
               {
                  "family_name": "Earhart",
                  "given_name": "Amelia",
                  #"loyalty_programme_accounts": [
                  #   {
                  #      "account_number": "12901014",
                  #      "airline_iata_code": "BA"
                  #   }
                  #],
                  "type": "adult",
               },
            ])
            # TODO: try adding in all my params to slices to do everything in a single call
            .slices([
               {
                  "origin": origin,
                  "destination": destination,
                  "departure_date": departure_date.strftime("%Y-%m-%d"),
               }
            ])
            .execute()
        )
    return reqfn()




# Iterate over the city pairs and search for flights
def get_flight_data(cities, start_date, end_date, airlines=['AA', 'UA', 'AM']):
    flights = []
    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for origin, destination in permutations(cities, 2):
            for single_date in pd.date_range(start_date, end_date, freq='D'):
                futures.append(
                    executor.submit(search_flights, origin, destination, single_date)
                )
    flight_datas = [future.result() for future in futures if future.result() is not None]
    for flight_data in flight_datas:
        # TODO can store id in db so we can actually book flights later
        for offer in flight_data.offers:
            # TODO or possible marketing carrier
            #if offer.owner.iata_code in airlines:
            if len(offer.slices) > 1:
                print("OFFER SLICES", len(offer.slices))
            flights.append(
                {
                    'Airline': offer.owner.name,
                    'Airline Code': offer.owner.iata_code,
                    'Cost': offer.total_amount,
                    'Origin': offer.slices[0].origin.iata_city_code,
                    'Destination': offer.slices[0].destination.iata_city_code,
                    'Departure Time': offer.slices[0].segments[0].departing_at,
                    'Arrival Time': offer.slices[0].segments[-1].arriving_at,
                    'Stops': len(offer.slices[0].segments) - 1,
            })
    return pd.DataFrame(flights)

def get_approx_flight_data(cities, airlines=['AA', 'UA', 'AM']):
    flights = pd.DataFrame()
    direct_flights = {city: get_airport_flights(city) for city in cities}
    for origin, destination in permutations(cities, 2):
        origin_direct_flights = direct_flights[origin][direct_flights[origin]['Destination'] == destination]
        flights = pd.concat([flights, origin_direct_flights])
        direct_flights_maximum_time = (origin_direct_flights['Arrival Time'] - origin_direct_flights['Departure Time']).max()
        flights = pd.concat([flights, find_1_stop_flights(origin, destination, direct_flights_maximum_time)])
    # TODO airline code should be part of find_1_stop_flights
    flights = flights[flights['Airline Code'].isin(airlines)]
    return flights

def create_solver_input(flights_data):
    model = cp_model.CpModel()

    # Create a variable for each flight
    flight_vars = {}
    flights_data['Departure Time'] = pd.to_datetime(flights_data['Departure Time'])
    for _, flight in flights_data.iterrows():
        flight_date = flight['Departure Time'].date()
        flight_id = f"{flight['Origin']}_{flight['Destination']}_{flight_date}"
        # Create a boolean variable for the flight
        flight_vars[flight_id] = model.NewBoolVar(flight_id)
    # Variables to represent if the traveler is in a certain city on a certain day
    days_in_city = {}
    cities = set()
    for _, flight in flights_data.iterrows():
        cities.add(flight['Origin'])
        cities.add(flight['Destination'])
    cities = pd.concat([flights_data['Origin'], flights_data['Destination']]).unique()
    dates = flights_data['Date'].unique()
    for city in cities:
        for d in dates:
            var_id = f"{city}_{d.day}"
            days_in_city[var_id] = model.NewBoolVar(var_id)

    # Constraint 1: Be in Mexico City for at least 7 days prior to December 20
    # TODO change city to city code here and in get_flight_data
    #mexico_city_days = [days_in_city[f"MEX_{d}"] for d in range(1, 5)]
    mexico_city_days = [days_in_city[f"MEX_{d}"] for d in range(13, 20)]
    #model.Add(sum(mexico_city_days) >= 3)
    model.Add(sum(mexico_city_days) >= 7)

    # Constraint 2: Be in Oaxaca for at least 5 days
    oaxaca_days = [days_in_city[f"OAX_{d}"] for d in range(1, 32)]
    model.Add(sum(oaxaca_days) >= 5)

    # Constraint 3: Be in Costa Rica for at least 5 days but not more than 7 days
    costa_rica_days = [days_in_city[f"SJO_{d}"] for d in range(1, 32)]
    model.Add(sum(costa_rica_days) >= 5)
    model.Add(sum(costa_rica_days) <= 7)

    # Constraint 4: Be in Cabo for at least 4 days but not more than 5 days between December 15 and December 30
    cabo_days = [days_in_city[f"SJD_{d}"] for d in range(15, 31)]
    #cabo_days = [days_in_city[f"SJD_{d}"] for d in range(1, 32)]
    model.Add(sum(cabo_days) >= 4)
    model.Add(sum(cabo_days) <= 5)

    # Additional constraints needed:
    # - Constraints to select flights from the specific airlines only
    # - Constraints to model the sequence of flights

    # Return the model and the dictionary of flight and city day variables
    return model, flight_vars, days_in_city


def link_flights_to_days(model, flights_data, flight_vars, days_in_city, start_city):
    # Create a mapping from dates to all flights on that date
    flights_on_date = {}
    for _, flight in flights_data.iterrows():
        date = flight['Date']
        if date not in flights_on_date:
            flights_on_date[date] = []
        flights_on_date[date].append(flight)

    for date, flights in flights_on_date.items():
        flight_ids = [f"{flight['Origin']}_{flight['Destination']}_{flight['Date']}"
                      for flight in flights]
        # one flight per day
        model.AddAtMostOne([flight_vars[flight_id] for flight_id in flight_ids])
        for flight, flight_id in zip(flights, flight_ids):
            # Generate IDs for flight variable and day in destination city
            next_day = (flight['Date'] + timedelta(days=1)).day
            day_in_destination_id = f"{flight['Destination']}_{next_day}"
            if day_in_destination_id in days_in_city:
                # Add constraint: if flight is taken, traveler is in the destination city the next day
                print(f"Implication: {flight_id} -> {day_in_destination_id}")
                model.AddImplication(flight_vars[flight_id], days_in_city[day_in_destination_id])

            day_in_origin_id = f"{flight['Origin']}_{flight['Date'].day}"
            if date.day > 1 and flight['Origin'] != start_city and day_in_origin_id in days_in_city:
                # For simplicity, assume the traveler must be in the city of departure on the day of the flight
                print(f"NOT {flight_id} OR NOT {day_in_origin_id}")
                model.AddBoolOr([flight_vars[flight_id].Not(), days_in_city[day_in_origin_id].Not()])
    dates = flights_data['Date'].unique()
    cities = pd.concat([flights_data['Origin'], flights_data['Destination']]).unique()

    start_city_day_0 = f"{start_city}_{dates[0].day}"
    start_city_day_n = f"{start_city}_{dates[-1].day}"
    print(f"{start_city_day_0} == 1")
    print(f"{start_city_day_n} == 1")
    model.Add(days_in_city[f'{start_city_day_0}'] == 1)
    model.Add(days_in_city[f'{start_city_day_n}'] == 1)
    for date in dates[1:-1]:
        start_city_day_i = f"{start_city}_{date.day}"
        print(f"{start_city_day_i} == 0")
        model.Add(days_in_city[f'{start_city_day_i}'] == 0)
        # can only be in 1 city at a time
        # TODO print this out as well
        model.AddAtMostOne([days_in_city[f"{city}_{date.day}"] for city in cities])
    for city in cities:
        if city != start_city:
            city_day_0 = f"{city}_{dates[0].day}"
            city_day_n = f"{city}_{dates[-1].day}"
            print(f"{city_day_0} == 0")
            print(f"{city_day_n} == 0")
            model.Add(days_in_city[f'{city_day_0}'] == 0)
            model.Add(days_in_city[f'{city_day_n}'] == 0)
        for date in dates[1:]:
            all_prev_flights_to_city = flights_data[
                (flights_data['Destination'] == city)
                & (flights_data['Date'] < date)
            ]
            flight_ids = [f"{flight['Origin']}_{flight['Destination']}_{flight['Date']}"
                          for _, flight in all_prev_flights_to_city.iterrows()]
            day_in_city_id = f"{city}_{date.day}"
            # if we didn't take a flight to this city before this day, we're not in the city on this day
            print(f"At least one of the following: {flight_ids}, {day_in_city_id}")
            model.AddBoolOr([flight_vars[flight_id] for flight_id in flight_ids] + [days_in_city[day_in_city_id].Not()])




def set_objective(model, flights_data, flight_vars):
    # Create the objective: Minimize the total cost of the selected flights
    total_cost = sum(int(flight['Cost']) * flight_vars[f"{flight['Origin']}_{flight['Destination']}_{flight['Date']}"]
                     for _, flight in flights_data.iterrows())
    #  total_cost = 0
    #  for _, flight in flights_data.iterrows():
        #  flight_date = flight['Date']
        #  flight_id = f"{flight['Origin']}_{flight['Destination']}_{flight_date}"
        #  total_cost += int(flight['Cost']) * flight_vars[flight_id]

    model.Minimize(total_cost)

def solve(model, flight_vars, days_in_city):
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(list(flight_vars.values()))
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.Solve(model, solution_printer)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for varname, var in flight_vars.items():
            value = solver.Value(var)
            if value:
                print(f"{varname}")
        for varname, var in days_in_city.items():
            value = solver.Value(var)
            if value:
                print(f"{varname}")
    else:
        print("No solution found.")

# TODO: duffel is too slow
if __name__ == '__main__':

    cities = ["LAX", "MEX", "OAX", "SJO", "SJD"]
    exclude_airports = ["ONT"]
    start_date, end_date = date(2023, 12, 1), date(2024, 1, 1)
    #  flights = get_flight_data(cities, start_date, end_date)
    #  flights.to_csv('flights.csv', index=False)
    #  flights = pd.read_csv('flights.csv')

    approx_flights = get_approx_flight_data(cities)
    approx_flights.to_csv('approx_flights.csv', index=False)
    approx_flights = pd.read_csv('approx_flights.csv')
    # ignore which flight is taken on which day
    #flights['Date'] = pd.to_datetime(flights['Departure Time']).dt.date
    # todo be more specific about origin airport
    #flights = flights[~flights.Origin.isin(exclude_airports)]
    #flights = flights.drop_duplicates(['Origin', 'Destination', 'Date'])

    # first just find when we'll be in each city ignoring flight times and costs,
    # assuming flights run every day
    approx_flights = approx_flights.drop_duplicates(['Origin', 'Destination'])
    #flights = flights[flights['Origin'].isin(['Los Angeles', 'Mexico City'])]
    #flights = flights[flights['Destination'].isin(['Los Angeles', 'Mexico City'])]
    model, flight_vars, days_in_city = create_solver_input(approx_flights)
    link_flights_to_days(model, approx_flights, flight_vars, days_in_city, "LAX")
    # TODO objective should be to minimize total travel time
    # and return an ordered list of options
    set_objective(model, approx_flights, flight_vars)
    solve(model, flight_vars, days_in_city)
    # TODO: then we should reduce search space, add cost information and solve again

