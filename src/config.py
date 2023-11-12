import os

DEFAULT_BUCKET = os.environ.get("BUCKET", "gpt-travel-planner-data")
DEFAULT_FLIGHTS_FILE = os.environ.get(
    "DEFAULT_FLIGHTS_FILE", "2023-11-11/flights.pickle"
)
DEFAULT_FLIGHT_PREFERENCES_FILE = os.environ.get(
    "DEFAULT_FLIGHT_PREFERENCES_FILE", "flight_preferences.pickle"
)
DEFAULT_PASSENGER_INFO_FILE = os.environ.get(
    "DEFAULT_PASSENGER_INFO_FILE", "passenger_info.pickle"
)
