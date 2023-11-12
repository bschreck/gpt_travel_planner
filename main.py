from flask import escape

import functions_framework
import pandas as pd

from flight_picker import (
    seed_data,
    find_and_select_flights,
    compound_offers_with_metrics_to_json,
    DesiredFlightLeg,
    UserFlightPreferences,
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

DEFAULT_BUCKET = os.environ.get("BUCKET", "gpt-travel-planner-data")
DEFAULT_FLIGHTS_FILE = os.environ.get(
    "DEFAULT_FLIGHTS_FILE", "2023-11-11/flights.pickle"
)


def parse_pick_flights_json(
    request_json: dict,
) -> tuple[str, list[DesiredFlightLeg], int]:
    """Parse the request JSON into the user ID, flight legs, and max flights.
    Args:
        request_json (dict): The request JSON.
    Returns:
        The user ID, flight legs, and max flights.
    """
    user_id = request_json["user_id"]
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
    return user_id, flight_legs, max_flights


def get_user(user_id: str) -> UserFlightPreferences:
    user, flight_legs = seed_data()
    return user


@functions_framework.http
def pick_flights(request):
    """Flight Picker HTTP Cloud Function
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    user_id, flight_legs, max_flights = parse_pick_flights_json(request_json)
    user = get_user(user_id)
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


@functions_framework.http
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
