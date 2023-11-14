from flask import escape

import functions_framework
import pandas as pd

from flight_picker import (
    seed_data,
    find_and_select_flights,
    compound_offers_with_metrics_to_json,
    DesiredFlightLeg,
    UserFlightPreferences,
    set_flight_preferences as internal_set_flight_preferences,
    get_flight_preferences as internal_get_flight_preferences,
    get_user,
    parse_flight_preferences,
    parse_passenger_info,
    set_passenger_info as internal_set_passenger_info,
    get_passenger_info as internal_get_passenger_info,
)
from scheduler import Scheduler, parse_schedule_trip_json
from src.get_flight_data import (
    get_daily_flights_crawl_multithreaded,
    build_flight_costs_from_remote_file,
    make_local_flight_costs_full,
    get_iata_codes_by_country
)
from src.utils import upload_file_to_gcs, cache_with_ttl
import pickle
import os
from dataclasses import dataclass
from config import (
    DEFAULT_BUCKET,
    DEFAULT_FLIGHTS_FILE,
    DEFAULT_FLIGHT_COSTS_FILE,
)


def parse_pick_flights_json(
    request_json: dict,
) -> tuple[str, list[DesiredFlightLeg], int]:
    """Parse the request JSON into the user ID, flight legs, and max flights.
    Args:
        request_json (dict): The request JSON.
    Returns:
        The passenger_name, flight legs, and max flights.
    """
    passenger_name = request_json["passenger_name"]
    flight_legs = [
        DesiredFlightLeg(
            origin=flight_leg["origin"],
            destination=flight_leg["destination"],
            earliest_departure_time=pd.Timestamp(flight_leg["earliest_departure_time"]),
            latest_arrival_time=pd.Timestamp(flight_leg["latest_arrival_time"]),
        )
        for flight_leg in request_json["flight_legs"]
    ]
    max_flights = request_json["max_flights"]
    return passenger_name, flight_legs, max_flights


def set_flight_preferences(request):
    passenger_name, passenger_prefs = parse_flight_preferences(
        request.get_json(silent=True)
    )
    if not passenger_name:
        return ("passenger_name is required", 400, {})
    resp = internal_set_flight_preferences(passenger_name, passenger_prefs)
    return (resp, 200, {})


def get_flight_preferences(request):
    passenger_name = request.args.get("passenger_name")
    if not passenger_name:
        passenger_name = request.get_json(silent=True).get("passenger_name", None)
    if not passenger_name:
        return ("passenger_name is required", 400, {})
    passenger_prefs = internal_get_flight_preferences(passenger_name)
    if passenger_prefs is None:
        return ("passenger_name not found", 404, {})
    return (passenger_prefs.to_json(), 200, {})


def set_passenger_info(request):
    passenger_name, passenger_info = parse_passenger_info(request.get_json(silent=True))
    if not passenger_name:
        return ("passenger_name is required", 400, {})
    resp = internal_set_passenger_info(passenger_name, passenger_info)
    return (resp, 200, {})


def get_passenger_info(request):
    passenger_name = request.args.get("passenger_name")
    if not passenger_name:
        passenger_name = request.get_json(silent=True).get("passenger_name", None)
    if not passenger_name:
        return ("passenger_name is required", 400, {})
    passenger_info = internal_get_passenger_info(passenger_name)
    if passenger_info is None:
        return ("passenger_name not found", 404, {})
    return (passenger_info.to_json(), 200, {})


def pick_flights(request):
    request_json = request.get_json(silent=True)
    passenger_name, flight_legs, max_flights = parse_pick_flights_json(request_json)
    user = get_user(passenger_name)
    flights = find_and_select_flights(
        user=user, flight_legs=flight_legs, max_flights_to_return=max_flights
    )
    print("got flights", flights)

    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    return (compound_offers_with_metrics_to_json(flights), 200, headers)


def schedule_trip(request, bucket=DEFAULT_BUCKET, flight_costs_file=DEFAULT_FLIGHT_COSTS_FILE, flights_file=DEFAULT_FLIGHTS_FILE):
    try:
        params = parse_schedule_trip_json(request.get_json(silent=True))
    except ValueError as e:
        return (f"Error parsing input: {e}", 400, {})
    if not os.path.exists(flight_costs_file):
        if not os.path.exists(flights_file):
            print("downloading and building flight costs")
            flight_costs = build_flight_costs_from_remote_file(
                params.bucket, params.filename, params.filename
            )
        print("making local all pairs flight costs")
        flight_costs = make_local_flight_costs_full(flights_file, flight_costs_file)
    else:
        print("loading local all pairs flight costs")
        with open(flight_costs_file, "rb") as f:
            flight_costs = pickle.load(f)
    print("building scheduler")
    try:
        scheduler = Scheduler(
            ndays=params.ndays,
            flight_costs=flight_costs,
            contiguous_sequence_constraints=params.contiguous_sequence_constraints,
            start_city=params.start_city,
            end_city=params.end_city,
            date_range_constraints=params.date_range_constraints,
            relevant_cities=params.relevant_cities,
            must_visits=params.must_visits,
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        breakpoint()
        return (f"Error building scheduler: {e}", 500, {})
    try:
        scheduler.create_model()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        breakpoint()
        return (f"Error creating model: {e}", 500, {})
    try:
        print("solving")
        scheduler.solve()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        breakpoint()
        return (f"Error solving model: {e}", 500, {})
    try:
        result_records = scheduler.get_result_flight_records()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        breakpoint()
        return (f"Error getting result flight records: {e}", 500, {})

    return (result_records, 200, {})



@functions_framework.http
def functions_entrypoint(request):
    function = request.args.get("function")
    if function is None:
        return ("function is required", 400, {})
    try:
        if function == "set_flight_preferences":
            return set_flight_preferences(request)
        elif function == "get_flight_preferences":
            return get_flight_preferences(request)
        elif function == "set_passenger_info":
            return set_passenger_info(request)
        elif function == "get_passenger_info":
            return get_passenger_info(request)
        elif function == "pick_flights":
            return pick_flights(request)
        elif function == "schedule_trip":
            return schedule_trip(request)
        elif function == "get_iata_codes_by_country":
            return get_iata_codes_by_country(request)
        else:
            return (f"function {function} not found", 404, {})
    except Exception as e:
        print(e)
        # print stacktrace
        import traceback

        print(traceback.format_exc())

        return (f"Error: {e}", 500, {})
