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
from collections import defaultdict
from duffel_api.http_client import ApiError
from requests import HTTPError
from pyflightdata import FlightData
from haversine import haversine, Unit
import networkx as nx
from heapq import heappush, heappop
from functools import cache
from utils import persist_to_file
from shift_scheduling_app import add_soft_sequence_constraint
from get_flight_data import build_flight_costs, build_flight_costs_from_remote_file
import pickle

@cache
def flight_data():
    f = FlightData()
    f.login(os.environ['FLIGHT_RADAR_EMAIL'], os.environ['FLIGHT_RADAR_PASSWORD'])
    return f

@cache
def get_flights_from_to(origin, destination):
    flights = flight_data().get_flights_from_to(origin, destination)
    # TODO
    return flights

@persist_to_file('airport_details.p')
def get_airport_details(airport_code):
    @retry(
        wait=wait_random_exponential(multiplier=1, max=64),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ValueError, HTTPError)),
    )
    def get_airport_details_with_retry():
        details = flight_data().get_airport_details(airport_code)
        if 'position' not in details or not isinstance(details['position'], dict):
            raise ValueError("bad return")
        return details
    return get_airport_details_with_retry()

@cache
def get_paginated_data(fn, limit=100):
    page = 1
    res = []
    page_res = None
    while page_res is None or len(page_res) >= limit:
        page_res = fn(page=page, limit=limit)
        res.extend(page_res)
        page += 1
    return res

@persist_to_file('airport_flights.p')
@retry(
    wait=wait_random_exponential(multiplier=1, max=64),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((ValueError, KeyError, HTTPError)),
)
def get_airport_flights(airport_code, arrivals=True):
    # TODO cache to disk
    details = get_airport_details(airport_code)
    lat,lon = details['position']['latitude'], details['position']['longitude']

    if arrivals:
        flights_raw = get_paginated_data(lambda page, limit: flight_data().get_airport_arrivals(airport_code, page=page, limit=limit))
    else:
        flights_raw = get_paginated_data(lambda page, limit: flight_data().get_airport_departures(airport_code, page=page, limit=limit))

    if len(flights_raw) == 0:
        raise ValueError("bad return")
    flights = []
    for flight_outer in flights_raw:
        flight = flight_outer['flight']
        if not isinstance(flight['airport'], dict):
            raise ValueError("bad return")

        if arrivals:
            if not isinstance(flight['airport']['origin'], dict):
                raise ValueError("bad return")
            if not isinstance(flight['airport']['origin']['position'], dict):
                raise ValueError("bad return")
            origin = flight['airport']['origin']['code']['iata']
            origin_latitude = flight['airport']['origin']['position']['latitude']
            origin_longitude = flight['airport']['origin']['position']['longitude']
            destination = airport_code
            dest_latitude = lat
            dest_longitude = lon
        else:
            if not isinstance(flight['airport']['destination'], dict):
                raise ValueError("bad return")
            if not isinstance(flight['airport']['destination']['position'], dict):
                raise ValueError("bad return")
            origin = airport_code
            destination = flight['airport']['destination']['code']['iata']
            dest_latitude = flight['airport']['destination']['position']['latitude']
            dest_longitude = flight['airport']['destination']['position']['longitude']
            origin_latitude = lat
            origin_longitude = lon

        owner_code = None
        owner_name = None
        if flight.get('Owner', {}).get('code', None) is not None:
            owner_code = flight['owner']['code']['iata']
            owner_name = flight['owner'].get('name', '')
        arrival_time = flight['time']['scheduled']['arrival']
        departure_time = flight['time']['scheduled']['departure']
        airline_name = None
        airline_code = None
        if isinstance(flight.get('airline', None), dict):
            airline_name = flight['airline']['name']
            airline_code = flight['airline']['code']['iata']
        else:
            continue
        distance = haversine(
            (origin_latitude, origin_longitude),
            (dest_latitude, dest_longitude),
            unit=Unit.MILES
        )
        flights.append(
            {
                'Airline': airline_name,
                'Airline Code': airline_code,
                'Owner': owner_name,
                'Owner Code': owner_code,
                'Cost': 0,
                'Origin': origin,
                'Destination': destination,
                'Departure Time': departure_time,
                'Arrival Time': arrival_time,
                'Travel Time': (arrival_time - departure_time)/3600,
                'Stops': 0,
                'Origin Latitude': origin_latitude,
                'Origin Longitude': origin_longitude,
                'Destination Latitude': dest_latitude,
                'Destination Longitude': dest_longitude,
                'Distance': distance,
            }
        )
    return pd.DataFrame(flights).sort_values('Distance')


def find_1_stop_flights(origin, destination, direct_flights_time, max_stops=2):
    # Priority queue for candidate flights, sorted by total travel time
    candidate_flights = []

    # Set to keep track of visited airports to prevent re-visiting
    visited_airports = set()
    added_flights = set()

    # Get initial flights from the origin
    origin_flights = get_airport_flights(origin, arrivals=False)
    for _, flight in origin_flights.iterrows():
        visited_airports.add(flight['Destination'])
        heappush(candidate_flights, (flight['Travel Time'], [flight.to_list()], [flight.to_dict()]))

    # List to store valid 1-stop connections
    valid_connections = []

    while candidate_flights:
        total_travel_time, _, flights = heappop(candidate_flights)
        last_flight = flights[-1]

        # Check if the current path can be completed with a flight to the destination
        if last_flight['Destination'] != destination:
            # Get connecting flights from the last airport
            connecting_flights = get_airport_flights(last_flight['Destination'], arrivals=False)
            for _, conn_flight in connecting_flights.iterrows():
                # Avoid cycles by skipping visited airports
                if conn_flight['Destination'] in visited_airports:
                    continue

                # Calculate layover time
                layover_time = max((conn_flight['Departure Time'] - last_flight['Arrival Time']) / 3600, 0)

                # Total travel time with layover
                new_total_time = total_travel_time + conn_flight['Travel Time'] + layover_time

                # Check the total travel time so far against the direct flight time
                if new_total_time <= direct_flights_time + 5 and layover_time >= 1:
                    # If this path is valid, add it to the list of valid connections
                    new_flights = flights + [conn_flight.to_dict()]
                    visited_airports.add(conn_flight['Destination'])
                    if len(new_flights) == max_stops and conn_flight['Destination'] != destination:
                        continue
                    heappush(candidate_flights, (new_total_time, [flight.to_list() for flight in new_flights], new_flights))

        else:
            # If we have reached the destination, calculate the total travel time
            # and check if the layover is at least 1 hour
            if len(flights) == max_stops:
                first_flight = flights[0]
                layover_time = (last_flight['Departure Time'] - first_flight['Arrival Time']) / 3600
                if layover_time >= 1:
                    # Save this connection as a valid option
                    valid_connections.append((total_travel_time, flights))

            # Stop adding new airports once we exceed the time limit
            if total_travel_time > direct_flights_time + 5:
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
        #  for v in self.__variables:
            #  print(f"{v}={self.Value(v)}", end=" ")
        #  print()

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
    direct_flights = {city: get_airport_flights(city, arrivals=False) for city in cities}
    for origin, destination in permutations(cities, 2):
        origin_direct_flights = direct_flights[origin][direct_flights[origin]['Destination'] == destination]
        flights = pd.concat([flights, origin_direct_flights])
        if len(origin_direct_flights) == 0:
            # estimate time by miles between airports, approx 400 mph avg speed.
            # TODO Should probably increase speed for longer flights

            origin_details = get_airport_details(origin)
            dest_details = get_airport_details(destination)
            distance = haversine(
                (origin_details['position']['latitude'], origin_details['position']['longitude']),
                (dest_details['position']['latitude'], dest_details['position']['longitude']),
                unit=Unit.MILES
            )
            direct_flights_maximum_time = distance / 400
        else:
            direct_flights_maximum_time = (origin_direct_flights['Arrival Time'] - origin_direct_flights['Departure Time']).max() / 3600
        valid_flights = find_1_stop_flights(origin, destination, direct_flights_maximum_time)
        breakpoint()
        flights = pd.concat([flights, pd.DataFrame(valid_flights[0][1])])
        breakpoint()
    # TODO airline code should be part of find_1_stop_flights
    flights = flights[flights['Airline Code'].isin(airlines)]
    return flights

def create_solver_input_simple(available_flights, start_day, end_day, max_visits):
    # available_flights is dict from origin to destination
    model = cp_model.CpModel()

    # Create a variable for each flight
    flight_vars = {}
    #  flights_data['Departure Time'] = pd.to_datetime(flights_data['Departure Time'])
    #  for _, flight in flights_data.iterrows():
        #  flight_date = flight['Departure Time'].date()
        #  flight_id = f"{flight['Origin']}_{flight['Destination']}_{flight_date}"
        #  # Create a boolean variable for the flight
        #  flight_vars[flight_id] = model.NewBoolVar(flight_id)
    for origin, destination in available_flights:
        for day in range(start_day, end_day+1):
            for ovisit in range(1, max_visits.get(origin,1)+1):
                for dvisit in range(1, max_visits.get(destination,1)+1):
                    flight_id = f"{origin}_{ovisit}_{destination}_{dvisit}_{day}"
                    flight_vars[flight_id] = model.NewBoolVar(flight_id)
    # Variables to represent if the traveler is in a certain city on a certain day
    days_in_city = {}
    #  cities = set()
    #  for _, flight in flights_data.iterrows():
        #  cities.add(flight['Origin'])
        #  cities.add(flight['Destination'])
    #cities = pd.concat([flights_data['Origin'], flights_data['Destination']]).unique()
    cities = set([origin for origin, _ in available_flights] + [destination for _, destination in available_flights])
    #dates = flights_data['Date'].unique()
    for city in cities:
        for d in range(start_day, end_day+1):
            for visit in range(1, max_visits.get(city,1)+1):
                var_id = f"{city}_{visit}_{d}"
                days_in_city[var_id] = model.NewBoolVar(var_id)

    #  # Constraint 1: Be in Mexico City for at least 7 days prior to December 20
    #  # TODO change city to city code here and in get_flight_data
    #  #mexico_city_days = [days_in_city[f"MEX_{d}"] for d in range(1, 5)]
    #  mexico_city_days = [days_in_city[f"MEX_{d}"] for d in range(13, 20)]
    #  #model.Add(sum(mexico_city_days) >= 3)
    #  model.Add(sum(mexico_city_days) >= 7)

    #  # Constraint 2: Be in Oaxaca for at least 5 days
    #  oaxaca_days = [days_in_city[f"OAX_{d}"] for d in range(1, 32)]
    #  model.Add(sum(oaxaca_days) >= 5)

    #  # Constraint 3: Be in Nicaragua for at least 5 days but not more than 7 days
    #  costa_rica_days = [days_in_city[f"MGA_{d}"] for d in range(1, 32)]
    #  model.Add(sum(costa_rica_days) >= 5)
    #  model.Add(sum(costa_rica_days) <= 7)

    #  # Constraint 4: Be in Cabo for at least 4 days but not more than 5 days between December 15 and December 30
    #  cabo_days = [days_in_city[f"SJD_{d}"] for d in range(15, 31)]
    #  #cabo_days = [days_in_city[f"SJD_{d}"] for d in range(1, 32)]
    #  model.Add(sum(cabo_days) >= 4)
    #  model.Add(sum(cabo_days) <= 5)

    # Additional constraints needed:
    # - Constraints to select flights from the specific airlines only
    # - Constraints to model the sequence of flights

    # Return the model and the dictionary of flight and city day variables
    return model, flight_vars, days_in_city


def link_flights_to_days(model, available_flights, start_date, end_date, flight_vars, days_in_city, start_city):
    # Create a mapping from dates to all flights on that date
    #  flights_on_date = {}
    #  for _, flight in flights_data.iterrows():
        #  date = flight['Date']
        #  if date not in flights_on_date:
            #  flights_on_date[date] = []
        #  flights_on_date[date].append(flight)

    dates = pd.date_range(start_date, end_date, freq='D')
    for date in dates:
    #for date, flights in flights_on_date.items():
        #  flight_ids = [f"{flight['Origin']}_{flight['Destination']}_{flight['Date']}"
                      #  for flight in flights]
        flight_ids = [f"{origin}_{destination}_{date.day}" for origin, destination in available_flights]
        # one flight per day
        model.AddAtMostOne([flight_vars[flight_id] for flight_id in flight_ids])
        #for flight, flight_id in zip(flights, flight_ids):
        for (origin, destination), flight_id in zip(available_flights, flight_ids):
            # Generate IDs for flight variable and day in destination city
            #next_day = (flight['Date'] + timedelta(days=1)).day
            next_day = (date + timedelta(days=1)).day
            #day_in_destination_id = f"{flight['Destination']}_{next_day}"
            day_in_destination_id = f"{destination}_{next_day}"
            if day_in_destination_id in days_in_city:
                # Add constraint: if flight is taken, traveler is in the destination city the next day
                print(f"Implication: {flight_id} -> {day_in_destination_id}")
                model.AddImplication(flight_vars[flight_id], days_in_city[day_in_destination_id])

            day_in_origin_id = f"{origin}_{date.day}"
            #day_in_origin_id = f"{flight['Origin']}_{flight['Date'].day}"
            #if date.day > 1 and flight['Origin'] != start_city and day_in_origin_id in days_in_city:
            if date.day > 1 and origin != start_city and day_in_origin_id in days_in_city:
                # For simplicity, assume the traveler must be in the city of departure on the day of the flight
                print(f"NOT {flight_id} OR NOT {day_in_origin_id}")
                model.AddBoolOr([flight_vars[flight_id].Not(), days_in_city[day_in_origin_id].Not()])
    #dates = flights_data['Date'].unique()
    #cities = pd.concat([flights_data['Origin'], flights_data['Destination']]).unique()
    cities = set([origin for origin, _ in available_flights] + [destination for _, destination in available_flights])

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
            #  all_prev_flights_to_city = flights_data[
                #  (flights_data['Destination'] == city)
                #  & (flights_data['Date'] < date)
            #  ]
            all_prev_flight_ids_to_city = {
                f"{origin}_{destination}_{d.day}"
                for origin, destination in available_flights
                for d in pd.date_range(start_date, date - timedelta(days=1), freq='D')
                if destination == city
            }
            #  flight_ids = [f"{flight['Origin']}_{flight['Destination']}_{flight['Date']}"
                          #  for _, flight in all_prev_flights_to_city.iterrows()]
            day_in_city_id = f"{city}_{date.day}"
            # if we didn't take a flight to this city before this day, we're not in the city on this day
            print(f"At least one of the following: {all_prev_flight_ids_to_city}, NOT {day_in_city_id}")
            model.AddBoolOr([flight_vars[flight_id] for flight_id in all_prev_flight_ids_to_city] + [days_in_city[day_in_city_id].Not()])

    model.Add(sum(flight_vars.values()) <= 7)




def set_objective(model, flight_hours, flight_vars, start_date, end_date):
    # Create the objective: Minimize the total cost of the selected flights
    #total_cost = sum(int(flight['Cost']) * flight_vars[f"{flight['Origin']}_{flight['Destination']}_{flight['Date']}"]
    #                 for _, flight in flights_data.iterrows())
    total_cost = sum(int(hours) * flight_vars[f"{o}_{d}_{date.day}"]
                     for (o, d),hours in flight_hours.items()
                     for date in pd.date_range(start_date, end_date, freq='D'))
    #  total_cost = 0
    #  for _, flight in flights_data.iterrows():
        #  flight_date = flight['Date']
        #  flight_id = f"{flight['Origin']}_{flight['Destination']}_{flight_date}"
        #  total_cost += int(flight['Cost']) * flight_vars[flight_id]

    model.Minimize(total_cost)

def solve(model, flight_vars, days_in_city, obj_bool_vars, obj_bool_coeffs, transitions, flight_transitions):
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_presolve = False  # Disables presolve to see the search log.
    solver.parameters.cp_model_probing_level = 0  # Disables probing to see the search log.
    #solver.parameters.log_presolve_changes = True
    #solver.parameters.log_infeasible_subsets = True
    #solver.parameters.trace_search  = True
    #solver.parameters.trace_propagation  = True
    #solution_printer = VarArraySolutionPrinter(list(flight_vars.values()))
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    # Enumerate all solutions.
    # solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.Solve(model, solution_printer)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        flights = []
        for varname, var in flight_vars.items():
            value = solver.Value(var)
            if value:
                origin, ovisit, destination, dvisit, date = varname.split('_')
                flights.append((origin,ovisit, destination,dvisit,int(date)))
        flights = sorted(flights, key=lambda x: x[-1])
        if len(flights) == 0:
            print("No flights found")
        for origin, ovisit, destination, dvisit, date in flights:
            print(f"Flight: {origin}#{ovisit} -> {destination}#{dvisit} on {date}")
        day_to_city = defaultdict(list)
        for varname, var in days_in_city.items():
            value = solver.Value(var)
            if value:
                city, visit, day = varname.split('_')
                day_to_city[int(day)].append((city, visit))
        for day in sorted(day_to_city.keys()):
            to_print = " || ".join(f"{day}: {city} #{visit}" for city, visit in day_to_city[day])
            print(to_print)
        for (city, visit), day_transitions in transitions.items():
            for i, transition in enumerate(day_transitions):
                day = i + 1
                print(f"Transition({city}#{visit} {day}->{day+1}): {solver.Value(transition)}")

        print("Penalties:")
        for i, var in enumerate(obj_bool_vars):
            if solver.BooleanValue(var):
                penalty = obj_bool_coeffs[i]
                if penalty > 0:
                    print("  %s violated, penalty=%i" % (var.Name(), penalty))
                else:
                    print("  %s fulfilled, gain=%i" % (var.Name(), -penalty))
    elif status == cp_model.INFEASIBLE:
        # The following will print the indices of the constraints that are part of the infeasibility proof
        print('Infeasible subsets:')
        print(solver.ResponseProto())
        #print(solver.ResponseProto().infeasible_constraints)
       # with open('model.pbtxt', 'w') as f:
       #     f.write(str(model.ModelProto()))
    else:
        print("No solution found.")

    print()
    print("Statistics")
    print("  - status          : %s" % solver.StatusName(status))
    print("  - conflicts       : %i" % solver.NumConflicts())
    print("  - branches        : %i" % solver.NumBranches())
    print("  - wall time       : %f s" % solver.WallTime())

def limit_contiguous_subsequences(model, days_in_city, max_visits, city, visit, must_visit):
    """
    Adds constraints to the model to limit the number of discrete contiguous subsequences.

    Args:
        model: The CP-SAT model.
        days_in_city: A list of BoolVar where each BoolVar represents presence in a city on a day.
        max_visits: The maximum allowed number of visits (discrete contiguous subsequences).
    """

    transitions = []
    num_days = len(days_in_city)

    # Identify transitions from 'not in city' to 'in city'
    for i in range(1, num_days):
        # If not in city on day i-1 and in city on day i, it's a new visit
        transition = model.NewBoolVar(f'transition({city},{visit},{i})')
        #  if city != 'LAX':
            #  breakpoint()
        model.AddBoolAnd([days_in_city[i-1].Not(), days_in_city[i]]).OnlyEnforceIf(transition)
        transitions.append(transition)

    # Count the number of visits and add constraint
    model.Add(sum(transitions) <= max_visits)
    if must_visit:
        model.Add(sum(transitions) >= 1)
    elif max_visits > 0:
        any_days_in_city = model.NewBoolVar(f'{city}_{visit}_any_days')
        model.AddMaxEquality(any_days_in_city, days_in_city)
        model.Add(sum(transitions) >= 1).OnlyEnforceIf(any_days_in_city)
    #if visit > 1:
    #    model.AddBoolXOr(days_in_city[i], transition.Not()).OnlyEnforceIf(transition)

    return transitions

def get_days_in_city_var(days_in_city, city, visit, date):
    return days_in_city[f"{city}_{visit}_{date}"]

def get_flight_var(flight_vars, origin, dest, day, ovisit, dvisit):
    return flight_vars[f"{origin}_{ovisit}_{dest}_{dvisit}_{day}"]

if __name__ == '__main__':

    #cities = ["LAX", "MEX", "OAX", "MGA", "SJD"]
    #exclude_airports = ["ONT"]
    start_date, end_date = date(2023, 12, 1), date(2024, 1, 1)
    #  flights = get_flight_data(cities, start_date, end_date)
    #  flights.to_csv('flights.csv', index=False)
    #  flights = pd.read_csv('flights.csv')

    #  approx_flights = get_approx_flight_data(cities)
    #  approx_flights.to_csv('approx_flights.csv', index=False)
    #  approx_flights = pd.read_csv('approx_flights.csv')
    #  breakpoint()
    #  # ignore which flight is taken on which day
    #  #flights['Date'] = pd.to_datetime(flights['Departure Time']).dt.date
    #  # todo be more specific about origin airport
    #  #flights = flights[~flights.Origin.isin(exclude_airports)]
    #  #flights = flights.drop_duplicates(['Origin', 'Destination', 'Date'])

    #  # first just find when we'll be in each city ignoring flight times and costs,
    #  # assuming flights run every day
    #  approx_flights = approx_flights.drop_duplicates(['Origin', 'Destination'])
    #  #flights = flights[flights['Origin'].isin(['Los Angeles', 'Mexico City'])]
    #  #flights = flights[flights['Destination'].isin(['Los Angeles', 'Mexico City'])]


    """
    Contiguous sequence constraints on each city (hard/soft min/max days in each)
    Disallow transitions: in_A_day_i, in_B_day_i+1, not flight from A to B day i
    Penalize unfavorable transitions (add a cost associated with each flight)
    Disallow 2 cities at once (for each day, sum of cities is 1)
    Disallow more than 1 flight per day (for each day, sum of flights is 1)
    Disallow more than X contiguous sequences in a city (default 1 - TODO figure out how to do more than 1)
        - for all pairs (i,j) of days separated by at least 1, from 1:-2 (first and last days must be in depot)
            for each day k in between
                disallow in_X_day_i, not(in_X_day_k), in_X_day_j
    First and last day must be in start city and end city
    """
    flight_costs = {
        ('LAX', 'MEX'): 3,
        ('MEX', 'LAX'): 3,
        ('MEX', 'OAX'): 1,
        ('OAX', 'MEX'): 1,
        ('LAX', 'MGA'): 5,
        ('MGA', 'LAX'): 5,
        ('MEX', 'MGA'): 3,
        ('MGA', 'MEX'): 3,
        ('SJD', 'LAX'): 2,
        ('LAX', 'SJD'): 2,
        ('SJD', 'MEX'): 2,
        ('MEX', 'SJD'): 2,
    }
    bucket = "gpt-travel-planner-data"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    flight_costs = build_flight_costs_from_remote_file(bucket, f'{today}/flights.pickle')
    #flight_costs = build_flight_costs_from_remote_file(bucket, f'flights.pickle')
    breakpoint()
    #with open('flights.pickle', 'rb') as f:
    #    flights = pickle.load(f)
    #flight_costs = build_flight_costs(flights)


    # (city, hard_min, soft_min, min_cost, hard_max, soft_max, max_cost, max_visits)
    #  contiguous_sequence_contraints = [
        #  ('MEX', 7, 10, 100, 10, 7, 100, 2),
        #  ('OAX', 5, 7, 100, 7,  7, 100, 1),
        #  ('MGA', 6, 7, 100, 7,  7, 100, 1),
        #  ('SJD', 3, 4, 100, 5,  4, 100, 1),
    #  ]

    contiguous_sequence_contraints = [
        ('SLC', 7, 10, 100, 30, 7, 100, 2),
        #('SEA', 5, 7, 100, 7,  7, 100, 1),
        #('ORD', 6, 7, 100, 7,  7, 100, 1),
        ('MEX', 3, 4, 100, 5,  4, 100, 1),
    ]
    start_city, end_city = 'LAX', 'LAX'
    relevant_cities = set([start_city, end_city] + [sc[0] for sc in contiguous_sequence_contraints])
    flight_costs = {
        (o, d): c
        for (o, d), c in flight_costs.items()
        if o in relevant_cities or d in relevant_cities
    }
    max_visits = {city: max_visit for city, _, _, _, _, _, _, max_visit in contiguous_sequence_contraints}

    ndays = 31

    model, flight_vars, days_in_city = create_solver_input_simple(flight_costs, 1, ndays, max_visits)

    cities = set([origin for origin, _ in flight_costs] + [destination for _, destination in flight_costs])
    # TODO don't hardcode month
    obj_bool_vars = []
    obj_bool_coeffs = []

    for sc in contiguous_sequence_contraints:
        city, hard_min, soft_min, min_cost, hard_max, soft_max, max_cost, max_visit = sc
        sequence = [get_days_in_city_var(days_in_city,city,1,d) for d in range(1, ndays+1)]
        # forbid sequences where the length of the longest contiguous sequence of city X is < hard_min or > hard_max
        # forbid sequences where the number of discrete contiguous sequence of city X is < hard_min or > hard_max
        # TODO: this should be at least one sequence following these constraints, but additional ones can be less than hard_min
        # and shouldn't be penalized (as much)
        variables, coeffs = add_soft_sequence_constraint(
            model,
            sequence,
            hard_min,
            soft_min,
            min_cost,
            soft_max,
            hard_max,
            max_cost,
            "contiguous_sequence_constraint(city %s #%d)" % (city, 1),
        )
        obj_bool_vars.extend(variables)
        obj_bool_coeffs.extend(coeffs)
        # have to be in city at least one day
        model.AddAtLeastOne(sequence)
        if max_visit > 1:
            for i in range(2, max_visit+1):
                hard_min = 1
                soft_min = 1
                sequence = [get_days_in_city_var(days_in_city,city,i,d) for d in range(1, ndays+1)]
                variables, coeffs = add_soft_sequence_constraint(
                    model,
                    sequence,
                    hard_min,
                    soft_min,
                    min_cost,
                    soft_max,
                    hard_max,
                    max_cost,
                    "contiguous_sequence_constraint(city %s #%d)" % (city, i),
                )
                obj_bool_vars.extend(variables)
                obj_bool_coeffs.extend(coeffs)

    # Disallow transitions: NOT(in_A_day_i ^ in_B_day_i+1 ^ not flight from A to B day i)
    # == OR(NOT(in_A_day_i), NOT(in_B_day_i+1), flight from A to B day i)
    flight_transitions = {}
    for origin, destination in permutations(cities, 2):
        origin_max_visits = max_visits.get(origin, 1)
        dest_max_visits = max_visits.get(destination, 1)
        for day in range(1, ndays):
            for ovisit in range(1, origin_max_visits+1):
                for dvisit in range(1, dest_max_visits+1):
                    origin_city_var = get_days_in_city_var(days_in_city, origin, ovisit, day)
                    dest_city_var = get_days_in_city_var(days_in_city, destination, dvisit, day+1)
                    transition = [
                        origin_city_var.Not(),
                        dest_city_var.Not(),
                    ]

                    if (origin, destination) not in flight_costs:
                        # Disallow transitions that don't have a flight
                        model.AddBoolOr(transition)
                    else:
                        # maybe something to do with LAX?

                        flight_var = get_flight_var(flight_vars, origin, destination, day, ovisit, dvisit)
                        cost = flight_costs[(origin, destination)]
                        #took_transition = model.NewBoolVar(f"took_transition({origin},{destination},{day})")
                        #model.AddBoolAnd([origin_city_var, dest_city_var]).OnlyEnforceIf(took_transition)

                        model.AddImplication(flight_var, origin_city_var)
                        model.AddImplication(flight_var, dest_city_var)

                        not_origin_or_not_dest = model.NewBoolVar(f'not_origin_{origin}_{day}_OR_not_dest_{destination}_{day+1})')

                        model.AddBoolOr([origin_city_var.Not(), dest_city_var.Not()]).OnlyEnforceIf(not_origin_or_not_dest)
                        model.AddImplication(flight_var.Not(), not_origin_or_not_dest)


                        #model.AddImplication(flight_var.Not(), not_dest_or_not_origin == 1)

                        #model.AddBoolOr([flight_var.Not(), origin_city_var, dest_city_var])
                        #model.AddBoolOr([flight_var, origin_city_var.Not()])
                        #model.AddBoolOr([flight_var, dest_city_var.Not()])


                        #model.AddBoolXOr([took_transition.Not(), flight_var])
                        obj_bool_vars.append(flight_var)
                        obj_bool_coeffs.append(cost)
                        #flight_transitions[(origin, destination, day)] = took_transition

    #Disallow 2 cities at once (for each day, sum of cities is 1)
    for day in range(1, ndays+1):
        model.AddExactlyOne([
            get_days_in_city_var(days_in_city,city, visit, day)
            for city in cities
            for visit in range(1, max_visits.get(city, 1)+1)
            #for visit in range(1,2)
        ])
    #Disallow more than 1 flight per day (for each day, sum of flights is 1)
    for day in range(1, ndays+1):
        flights_on_day = [
            get_flight_var(flight_vars, o, d, day, ovisit, dvisit)
            for o, d in flight_costs
            for ovisit in range(1, max_visits.get(o, 1)+1)
            for dvisit in range(1, max_visits.get(d, 1)+1)
        ]
        model.Add(sum(flights_on_day) <= 1)

    #Disallow more than X contiguous sequences in a city
    transitions = {}
    for city in cities:
        if city == start_city:
            max_visit = 0
            visits = [1]
        elif city == end_city:
            max_visit = 0
            visits = [1]
        else:
            max_visit = 1
            visits = range(1, max_visits.get(city, 1)+1)
        for visit in visits:
            # TODO this can be parameterized if user wants to visit a city at least N times
            must_visit = visit == 1 and city != start_city
            days_in_city_vars = [
                get_days_in_city_var(days_in_city,city,visit,day)
                for day in range(1,ndays+1)
            ]
            transitions[(city, visit)] = limit_contiguous_subsequences(
                model,
                days_in_city_vars,
                max_visit,
                city,
                visit,
                must_visit
            )


    model.Add(get_days_in_city_var(days_in_city,start_city,1,1) == 1)
    model.Add(get_days_in_city_var(days_in_city,end_city,1,ndays) == 1)

    model.Minimize(
        sum(obj_bool_vars[i] * obj_bool_coeffs[i] for i in range(len(obj_bool_vars)))
    )

    solve(model, flight_vars, days_in_city, obj_bool_vars, obj_bool_coeffs, transitions, flight_transitions)






    #link_flights_to_days(model, available_flights, start_date, end_date, flight_vars, days_in_city, "LAX")
    # TODO objective should be to minimize total travel time
    # and return an ordered list of options
    #set_objective(model, flight_hours, flight_vars, start_date, end_date)
    # TODO: then we should reduce search space, add cost information and solve again

