import pytest
import subprocess
import requests
import time

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


def test_seed_data():
    data = {
        "user_id": "test_user",
        "flight_legs": [
            {
                "origin": "SFO",
                "destination": "LAX",
                "earliest_departure_time": "2024-01-01 07:00:00",
                "latest_arrival_time": "2024-01-01 22:00:00",
            }
        ],
        "max_flights": 1,
    }
    response = requests.post(
        "https://us-west1-travel-planner-404820.cloudfunctions.net/python-http-function",
        # "http://localhost:8080",
        json=data,
    )
    print(response.text)
    assert response.status_code == 200
    print(response.json())
    assert response.json() == [
        {
            "display_metrics": {
                "is_preferred_airline": [True],
                "layover_durations": [0],
                "max_total_duration": 1.5833333333333333,
                "nstops": 0,
                "time_of_days": ["morning"],
                "total_cost": 79.21,
            },
            "metrics": {
                "layover_duration": 0.0,
                "nstops": 1.0,
                "preferred_airline_ratio": 1.0,
                "time_of_day": 1.0,
                "total_cost": 0.960395,
                "total_duration": 0.9208333333333334,
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
                }
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
    response = requests.post("http://localhost:8080", json=data)
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
