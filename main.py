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
from scheduler import Scheduler, ContiguousSequenceConstraint, DateRangeConstraint
from src.get_flight_data import (
    get_daily_flights_crawl_multithreaded,
    build_flight_costs_from_remote_file,
)
from src.utils import upload_file_to_gcs
import pickle
import os
from dataclasses import dataclass
from config import DEFAULT_BUCKET, DEFAULT_FLIGHTS_FILE


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
    internal_set_flight_preferences(passenger_name, passenger_prefs)
    return ("", 200, {})


def get_flight_preferences(request):
    passenger_name = request.args.get("passenger_name")
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
    internal_set_passenger_info(passenger_name, passenger_info)
    return ("", 200, {})


def get_passenger_info(request):
    passenger_name = request.args.get("passenger_name")
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


@dataclass
class ScheduleTripParams:
    start_city: str
    ndays: int
    bucket: str = DEFAULT_BUCKET
    filename: str = DEFAULT_FLIGHTS_FILE
    contiguous_sequence_constraints: list[ContiguousSequenceConstraint] | None = None
    date_range_constraints: list[DateRangeConstraint] | None = None
    end_city: str | None = None


def parse_schedule_trip_json(request_json: dict) -> ScheduleTripParams:
    start_city = request_json["start_city"]
    end_city = request_json.get("end_city", None)
    ndays = request_json["ndays"]
    bucket = request_json.get("bucket", DEFAULT_BUCKET)
    filename = request_json.get("filename", DEFAULT_FLIGHTS_FILE)
    contiguous_sequence_constraints = []
    for sc_raw in request_json.get("contiguous_sequence_constraints", []):
        contiguous_sequence_constraints.append(
            ContiguousSequenceConstraint(
                city=sc_raw["city"],
                hard_min=sc_raw["hard_min"],
                soft_min=sc_raw["soft_min"],
                soft_max=sc_raw["soft_max"],
                hard_max=sc_raw["hard_max"],
                min_cost=sc_raw.get("min_cost", 100),
                max_cost=sc_raw.get("max_cost", 100),
                max_visits=sc_raw.get("max_visits", 1),
            )
        )
    date_range_constraints = []
    for drc_raw in request_json.get("date_range_constraints", []):
        date_range_constraints.append(
            DateRangeConstraint(
                city=drc_raw["city"],
                min_start_day=drc_raw.get("min_start_day", None),
                max_start_day=drc_raw.get("max_start_day", None),
                min_end_day=drc_raw.get("min_end_day", None),
                max_end_day=drc_raw.get("max_end_day", None),
                visit=drc_raw.get("visit", 1),
            )
        )

    return ScheduleTripParams(
        start_city=start_city,
        end_city=end_city,
        ndays=ndays,
        bucket=bucket,
        filename=filename,
        contiguous_sequence_constraints=contiguous_sequence_constraints,
        date_range_constraints=date_range_constraints,
    )


def schedule_trip(request):
    params = parse_schedule_trip_json(request.get_json(silent=True))
    print("building flight costs")
    flight_costs = build_flight_costs_from_remote_file(
        params.bucket, params.filename, params.filename
    )
    print("building scheduler")
    scheduler = Scheduler(
        ndays=params.ndays,
        flight_costs=flight_costs,
        contiguous_sequence_constraints=params.contiguous_sequence_constraints,
        start_city=params.start_city,
        end_city=params.end_city,
        date_range_constraints=params.date_range_constraints,
    )
    try:
        scheduler.create_model()
    except Exception as e:
        return (f"Error creating model: {e}", 500, {})
    try:
        print("solving")
        scheduler.solve()
    except Exception as e:
        return (f"Error solving model: {e}", 500, {})
    result_records = scheduler.get_result_flight_records()

    return (result_records, 200, {})


@functions_framework.http
def functions_entrypoint(request):
    function = request.args.get("function")
    if function is None:
        return ("function is required", 400, {})
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
    else:
        return (f"function {function} not found", 404, {})
