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
    f.login(os.environ["FLIGHT_RADAR_EMAIL"], os.environ["FLIGHT_RADAR_PASSWORD"])
    return f


@cache
def get_flights_from_to(origin, destination):
    flights = flight_data().get_flights_from_to(origin, destination)
    # TODO
    return flights


@persist_to_file("airport_details.p")
def get_airport_details(airport_code):
    @retry(
        wait=wait_random_exponential(multiplier=1, max=64),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ValueError, HTTPError)),
    )
    def get_airport_details_with_retry():
        details = flight_data().get_airport_details(airport_code)
        if "position" not in details or not isinstance(details["position"], dict):
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


@persist_to_file("airport_flights.p")
@retry(
    wait=wait_random_exponential(multiplier=1, max=64),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((ValueError, KeyError, HTTPError)),
)
def get_airport_flights(airport_code, arrivals=True):
    # TODO cache to disk
    details = get_airport_details(airport_code)
    lat, lon = details["position"]["latitude"], details["position"]["longitude"]

    if arrivals:
        flights_raw = get_paginated_data(
            lambda page, limit: flight_data().get_airport_arrivals(
                airport_code, page=page, limit=limit
            )
        )
    else:
        flights_raw = get_paginated_data(
            lambda page, limit: flight_data().get_airport_departures(
                airport_code, page=page, limit=limit
            )
        )

    if len(flights_raw) == 0:
        raise ValueError("bad return")
    flights = []
    for flight_outer in flights_raw:
        flight = flight_outer["flight"]
        if not isinstance(flight["airport"], dict):
            raise ValueError("bad return")

        if arrivals:
            if not isinstance(flight["airport"]["origin"], dict):
                raise ValueError("bad return")
            if not isinstance(flight["airport"]["origin"]["position"], dict):
                raise ValueError("bad return")
            origin = flight["airport"]["origin"]["code"]["iata"]
            origin_latitude = flight["airport"]["origin"]["position"]["latitude"]
            origin_longitude = flight["airport"]["origin"]["position"]["longitude"]
            destination = airport_code
            dest_latitude = lat
            dest_longitude = lon
        else:
            if not isinstance(flight["airport"]["destination"], dict):
                raise ValueError("bad return")
            if not isinstance(flight["airport"]["destination"]["position"], dict):
                raise ValueError("bad return")
            origin = airport_code
            destination = flight["airport"]["destination"]["code"]["iata"]
            dest_latitude = flight["airport"]["destination"]["position"]["latitude"]
            dest_longitude = flight["airport"]["destination"]["position"]["longitude"]
            origin_latitude = lat
            origin_longitude = lon

        owner_code = None
        owner_name = None
        if flight.get("Owner", {}).get("code", None) is not None:
            owner_code = flight["owner"]["code"]["iata"]
            owner_name = flight["owner"].get("name", "")
        arrival_time = flight["time"]["scheduled"]["arrival"]
        departure_time = flight["time"]["scheduled"]["departure"]
        airline_name = None
        airline_code = None
        if isinstance(flight.get("airline", None), dict):
            airline_name = flight["airline"]["name"]
            airline_code = flight["airline"]["code"]["iata"]
        else:
            continue
        distance = haversine(
            (origin_latitude, origin_longitude),
            (dest_latitude, dest_longitude),
            unit=Unit.MILES,
        )
        flights.append(
            {
                "Airline": airline_name,
                "Airline Code": airline_code,
                "Owner": owner_name,
                "Owner Code": owner_code,
                "Cost": 0,
                "Origin": origin,
                "Destination": destination,
                "Departure Time": departure_time,
                "Arrival Time": arrival_time,
                "Travel Time": (arrival_time - departure_time) / 3600,
                "Stops": 0,
                "Origin Latitude": origin_latitude,
                "Origin Longitude": origin_longitude,
                "Destination Latitude": dest_latitude,
                "Destination Longitude": dest_longitude,
                "Distance": distance,
            }
        )
    return pd.DataFrame(flights).sort_values("Distance")


def find_1_stop_flights(origin, destination, direct_flights_time, max_stops=2):
    # Priority queue for candidate flights, sorted by total travel time
    candidate_flights = []

    # Set to keep track of visited airports to prevent re-visiting
    visited_airports = set()
    added_flights = set()

    # Get initial flights from the origin
    origin_flights = get_airport_flights(origin, arrivals=False)
    for _, flight in origin_flights.iterrows():
        visited_airports.add(flight["Destination"])
        heappush(
            candidate_flights,
            (flight["Travel Time"], [flight.to_list()], [flight.to_dict()]),
        )

    # List to store valid 1-stop connections
    valid_connections = []

    while candidate_flights:
        total_travel_time, _, flights = heappop(candidate_flights)
        last_flight = flights[-1]

        # Check if the current path can be completed with a flight to the destination
        if last_flight["Destination"] != destination:
            # Get connecting flights from the last airport
            connecting_flights = get_airport_flights(
                last_flight["Destination"], arrivals=False
            )
            for _, conn_flight in connecting_flights.iterrows():
                # Avoid cycles by skipping visited airports
                if conn_flight["Destination"] in visited_airports:
                    continue

                # Calculate layover time
                layover_time = max(
                    (conn_flight["Departure Time"] - last_flight["Arrival Time"])
                    / 3600,
                    0,
                )

                # Total travel time with layover
                new_total_time = (
                    total_travel_time + conn_flight["Travel Time"] + layover_time
                )

                # Check the total travel time so far against the direct flight time
                if new_total_time <= direct_flights_time + 5 and layover_time >= 1:
                    # If this path is valid, add it to the list of valid connections
                    new_flights = flights + [conn_flight.to_dict()]
                    visited_airports.add(conn_flight["Destination"])
                    if (
                        len(new_flights) == max_stops
                        and conn_flight["Destination"] != destination
                    ):
                        continue
                    heappush(
                        candidate_flights,
                        (
                            new_total_time,
                            [flight.to_list() for flight in new_flights],
                            new_flights,
                        ),
                    )

        else:
            # If we have reached the destination, calculate the total travel time
            # and check if the layover is at least 1 hour
            if len(flights) == max_stops:
                first_flight = flights[0]
                layover_time = (
                    last_flight["Departure Time"] - first_flight["Arrival Time"]
                ) / 3600
                if layover_time >= 1:
                    # Save this connection as a valid option
                    valid_connections.append((total_travel_time, flights))

            # Stop adding new airports once we exceed the time limit
            if total_travel_time > direct_flights_time + 5:
                break

    # Sort the valid connections by total travel time
    valid_connections.sort(key=lambda x: x[0])

    return valid_connections
