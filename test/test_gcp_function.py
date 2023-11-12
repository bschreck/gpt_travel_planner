import pytest
import subprocess
import requests
import time
import datetime
from flight_picker import (
    parse_flight_preferences,
    set_flight_preferences,
    get_flight_preferences,
    parse_passenger_info,
    set_passenger_info,
    get_passenger_info,
    PassengerInfo,
    User,
    UserFlightPreferences,
)

# TODO
# @pytest.fixture(scope="module", autouse=True)
# def start_gcp_function():
#    process = subprocess.Popen([
#        "functions-framework-python",
#         "--target",
#         "pick_flights",
#        "--source",
#        "src/gcp_function.py"
#    ],
#    stdout=subprocess.PIPE,
#    stderr=subprocess.PIPE,
#    shell=False, text=True)
#    # Wait for the server to start
#    output = False
#    timeout = 10  # Timeout in seconds
#    start_time = time.time()
#    while time.time() - start_time < timeout:
#        line = process.stdout.readline()
#        err = process.stderr.readline()
#        if err:
#            raise RuntimeError(f"Error starting process: {err}")
#        if line:
#            output = True
#            break
#        if process.poll() is not None:
#            # Process ended before any output
#            raise RuntimeError("Process terminated without output")
#
#    if not output:
#        raise RuntimeError("No output received from process within timeout period")
#    print(f"Started process with PID {process.pid}")
#    print(f"Output: {line}")
#
#
#    yield
#    # Cleanup: Terminate the process
#    process.terminate()
#    try:
#        process.wait(timeout=5)  # Wait for the process to terminate
#    except subprocess.TimeoutExpired:
#        process.kill()  # Force kill if it's not terminating
#        process.wait()  # Wait for force kill to complete


def test_pick_flights():
    passenger_info = PassengerInfo(
        name="John Doe",
        date_of_birth=datetime.date(1990, 1, 1),
        phone_number="123-456-7890",
        email="john@example.com",
        age=30,
    )
    flight_preferences = {
        "time_of_day_order": [
            "morning",
            "afternoon",
            "evening",
            "red_eye",
            "early_morning",
        ],
        "hard_max_cost": 500.0,
        "soft_max_cost": 400.0,
        "single_leg_hard_max_cost": 250.0,
        "single_leg_soft_max_cost": 200.0,
        "soft_max_duration": 14400,
        "hard_max_duration": 21600,
        "soft_max_stops": 2,
        "hard_max_stops": 3,
        "soft_min_layover_duration": 1800,
        "hard_min_layover_duration": 3600,
        "soft_max_layover_duration": 7200,
        "hard_max_layover_duration": 10800,
        "airline_preferences": {"American Airlines": 1, "United Airlines": 2},
        "seat_class_prefernces": {
            "economy": 1,
            "premium_economy": 2,
            "business": 3,
            "first": 4,
        },
        "seat_location_preference": "window",
        "seat_location_row_preference": "front",
        "desires_extra_legroom": True,
        "total_cost_weight": 0.4,
        "total_duration_weight": 0.2,
        "preferred_airline_ratio_weight": 0.05,
        "time_of_day_weight": 0.15,
        "layover_duration_weight": 0.1,
        "nstops_weight": 0.1,
    }

    set_passenger_info("John Doe", passenger_info)
    data = {"passenger_name": "John Doe", "preferences": flight_preferences}

    set_flight_preferences(*parse_flight_preferences(data))

    data = {
        "passenger_name": "John Doe",
        "flight_legs": [
            {
                "origin": "SFO",
                "destination": "LAX",
                "earliest_departure_time": "2024-01-01 07:00:00",
                "latest_arrival_time": "2024-01-01 22:00:00",
            },
            {
                "origin": "LAX",
                "destination": "MEX",
                "earliest_departure_time": "2024-01-07 04:00:00",
                "latest_arrival_time": "2024-01-07 22:00:00",
            },
            {
                "origin": "MEX",
                "destination": "SFO",
                "earliest_departure_time": "2024-01-10 04:00:00",
                "latest_arrival_time": "2024-01-10 22:00:00",
            },
        ],
        "max_flights": 1,
    }
    response = requests.post("http://localhost:8080?function=pick_flights", json=data)
    assert response.status_code == 200
    assert response.json() == [
        {
            "display_metrics": {
                "is_preferred_airline": [true, true, true],
                "layover_durations": [0, 0, 0],
                "max_total_duration": 5.916666666666667,
                "nstops": 0,
                "time_of_days": ["morning", "early_morning", "early_morning"],
                "total_cost": 393.36,
            },
            "metrics": {
                "layover_duration": 0.0,
                "nstops": 1.0,
                "preferred_airline_ratio": 1.0,
                "time_of_day": 0.4666666666666667,
                "total_cost": 0.75415,
                "total_duration": 0.9010416666666666,
            },
            "offers": [
                {
                    "slices": [
                        {
                            "segments": [
                                {
                                    "arriving_at": "2024-01-01T10:00:00",
                                    "departing_at": "2024-01-01T08:25:00",
                                    "destination": "LAX",
                                    "duration": 5700.0,
                                    "marketing_carrier": "United Airlines",
                                    "operating_carrier": "United Airlines",
                                    "origin": "SFO",
                                }
                            ]
                        }
                    ],
                    "total_amount": 79.21,
                },
                {
                    "slices": [
                        {
                            "segments": [
                                {
                                    "arriving_at": "2024-01-07T10:33:00",
                                    "departing_at": "2024-01-07T04:38:00",
                                    "destination": "MEX",
                                    "duration": 21300.0,
                                    "marketing_carrier": "American Airlines",
                                    "operating_carrier": "American Airlines",
                                    "origin": "LAX",
                                }
                            ]
                        }
                    ],
                    "total_amount": 143.06,
                },
                {
                    "slices": [
                        {
                            "segments": [
                                {
                                    "arriving_at": "2024-01-10T09:35:00",
                                    "departing_at": "2024-01-10T06:58:00",
                                    "destination": "SFO",
                                    "duration": 9420.0,
                                    "marketing_carrier": "American Airlines",
                                    "operating_carrier": "American Airlines",
                                    "origin": "MEX",
                                }
                            ]
                        }
                    ],
                    "total_amount": 171.09,
                },
            ],
        }
    ]


def test_schedule_trip():
    data = {
        "start_city": "LAX",
        "ndays": 31,
        "contiguous_sequence_constraints": [
            {
                "city": "MEX",
                "hard_min": 7,
                "soft_min": 10,
                "hard_max": 10,
                "soft_max": 7,
                "max_visits": 2,
            },
            {"city": "OAX", "hard_min": 5, "soft_min": 7, "hard_max": 7, "soft_max": 7},
            {"city": "MGA", "hard_min": 6, "soft_min": 7, "hard_max": 7, "soft_max": 7},
            {"city": "SJD", "hard_min": 3, "soft_min": 4, "hard_max": 5, "soft_max": 4},
        ],
        "date_range_constraints": [
            {"city": "MEX", "min_start_day": 2, "max_end_day": 20},
            {
                "city": "MGA",
                "min_start_day": 16,
                "max_start_day": 17,
                "min_end_day": 23,
                "max_end_day": 24,
            },
        ],
    }
    response = requests.post("http://localhost:8080?function=schedule_trip", json=data)
    assert response.status_code == 200
    # TODO result in non-deterministic
    assert response.json() == [
        {"day": 1, "destination": "MEX", "origin": "LAX"},
        {"day": 2, "destination": "OAX", "origin": "MEX"},
        {"day": 9, "destination": "MEX", "origin": "OAX"},
        {"day": 16, "destination": "MGA", "origin": "MEX"},
        {"day": 23, "destination": "SJD", "origin": "MGA"},
        {"day": 27, "destination": "LAX", "origin": "SJD"},
    ]


# TODO use random test file
def test_get_flight_preferences():
    prefs = {
        "time_of_day_order": [
            "morning",
            "afternoon",
            "evening",
            "red_eye",
            "early_morning",
        ],
        "hard_max_cost": 500.0,
        "soft_max_cost": 400.0,
        "single_leg_hard_max_cost": 250.0,
        "single_leg_soft_max_cost": 200.0,
        "soft_max_duration": 14400,
        "hard_max_duration": 21600,
        "soft_max_stops": 2,
        "hard_max_stops": 3,
        "soft_min_layover_duration": 1800,
        "hard_min_layover_duration": 3600,
        "soft_max_layover_duration": 7200,
        "hard_max_layover_duration": 10800,
        "airline_preferences": {"American Airlines": 1, "United Airlines": 2},
        "seat_class_prefernces": {
            "economy": 1,
            "premium_economy": 2,
            "business": 3,
            "first": 4,
        },
        "seat_location_preference": "window",
        "seat_location_row_preference": "front",
        "desires_extra_legroom": True,
        "total_cost_weight": 0.4,
        "total_duration_weight": 0.2,
        "preferred_airline_ratio_weight": 0.05,
        "time_of_day_weight": 0.15,
        "layover_duration_weight": 0.1,
        "nstops_weight": 0.1,
    }
    data = {"passenger_name": "test_user", "preferences": prefs}

    passenger, flight_prefs = parse_flight_preferences(data)
    set_flight_preferences(passenger, flight_prefs)
    received_flight_prefs = get_flight_preferences(passenger)
    assert received_flight_prefs == flight_prefs

    update_costs = {
        "total_cost_weight": 0.5,
        "total_duration_weight": 0.1,
        "preferred_airline_ratio_weight": 0.1,
        "time_of_day_weight": 0.1,
        "layover_duration_weight": 0.1,
        "nstops_weight": 0.1,
    }
    data = {"passenger_name": "test_user", "preferences": update_costs}
    passenger, flight_prefs = parse_flight_preferences(data)
    set_flight_preferences(passenger, flight_prefs)
    received_flight_prefs = get_flight_preferences(passenger)
    assert received_flight_prefs.soft_max_stops == 2
    assert received_flight_prefs.total_cost_weight == 0.5
    assert received_flight_prefs.total_duration_weight == 0.1
    assert received_flight_prefs.preferred_airline_ratio_weight == 0.1
    assert received_flight_prefs.time_of_day_weight == 0.1
    assert received_flight_prefs.layover_duration_weight == 0.1
    assert received_flight_prefs.nstops_weight == 0.1

    response = requests.post(
        "http://localhost:8080?function=set_flight_preferences", json=data
    )
    assert response.status_code == 200
    response = requests.get(
        "http://localhost:8080?function=get_flight_preferences&passenger_name=test_user"
    )
    assert response.status_code == 200


# TODO use random test file
def test_get_user_info():
    passenger_info = PassengerInfo(
        name="John Doe",
        date_of_birth=datetime.date(1990, 1, 1),
        phone_number="123-456-7890",
        email="john@example.com",
        age=30,
    )
    json_data = passenger_info.to_json()
    new_passenger_info = PassengerInfo.from_json(json_data)

    passenger_info2 = PassengerInfo(
        name="Jane Doe",
        date_of_birth=datetime.date(1992, 2, 2),
        phone_number="098-765-4321",
        email="jane@example.com",
        age=29,
    )
    merged_passenger_info = passenger_info.merge(passenger_info2)

    info = {
        "name": "John Doe",
        "date_of_birth": "1990-01-01",
        "phone_number": "123-456-7890",
        "ptype": "adult",
        "age": 30,
    }
    data = {"passenger_name": "test_user", "passenger_info": info}

    passenger, parsed_info = parse_passenger_info(data)
    set_passenger_info(passenger, parsed_info)
    received_flight_prefs = get_passenger_info(passenger)
    assert received_flight_prefs == parsed_info

    update = {"phone_number": "555-555-5555"}
    data = {"passenger_name": "test_user", "passenger_info": update}

    passenger, parsed_info = parse_passenger_info(data)
    set_passenger_info(passenger, parsed_info)
    received_info = get_passenger_info(passenger)
    assert received_info.phone_number == "555-555-5555"
    assert received_info.date_of_birth == datetime.date(1990, 1, 1)

    response = requests.post(
        "http://localhost:8080?function=set_passenger_info", json=data
    )
    assert response.status_code == 200
    response = requests.get(
        "http://localhost:8080?function=get_passenger_info&passenger_name=test_user"
    )
    assert response.status_code == 200
