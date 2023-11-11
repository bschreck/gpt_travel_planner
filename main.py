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
def parse_request_json(request_json: dict) -> tuple[str, list[DesiredFlightLeg], int]:
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

    user_id, flight_legs, max_flights = parse_request_json(request_json)
    user = get_user(user_id)
    print("got user")
    flights = find_and_select_flights(
        user=user,
        flight_legs=flight_legs,
        max_flights_to_return=max_flights,
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
