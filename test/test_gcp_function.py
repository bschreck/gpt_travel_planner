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
