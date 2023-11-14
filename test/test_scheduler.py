from scheduler import Scheduler, parse_schedule_trip_json
from get_flight_data import build_flight_costs
import pytest
import pickle

FLIGHT_COSTS_LOCAL_FILE = "flights_local.pickle"


@pytest.fixture(scope="module")
def flight_costs():
    with open(FLIGHT_COSTS_LOCAL_FILE, "rb") as f:
        flight_costs = pickle.load(f)
    return flight_costs


@pytest.mark.parametrize(
    "raw_args,expected",
    [
        (
            {
                "start_city": "LAX",
                "ndays": 31,
                "relevant_cities": ["MGA", "OAX"],
                "contiguous_sequence_constraints": [
                    {
                        "city": "MEX",
                        "soft_min": 5,
                        "hard_min": 4,
                        "soft_max": 8,
                        "hard_max": 10,
                        "max_visits": 1,
                    },
                    {
                        "city": "SJD",
                        "soft_min": 3,
                        "hard_min": 2,
                        "soft_max": 5,
                        "hard_max": 6,
                        "max_visits": 1,
                    },
                ],
                "date_range_constraints": [
                    {
                        "city": "MEX",
                        "min_start_day": 1,
                        "max_start_day": 13,
                        "min_end_day": 5,
                        "max_end_day": 20,
                        "visit": 1,
                    },
                    {
                        "city": "SJD",
                        "min_start_day": 15,
                        "max_start_day": 28,
                        "min_end_day": 18,
                        "max_end_day": 31,
                        "visit": 1,
                    },
                ],
            },
            [
                {"origin": "LAX", "destination": "MEX", "day": 1},
                {"origin": "MEX", "destination": "OAX", "day": 10},
                {"origin": "OAX", "destination": "SJD", "day": 14},
                {"origin": "SJD", "destination": "MGA", "day": 20},
                {"origin": "MGA", "destination": "LAX", "day": 30},
            ],
        )
    ],
)
def test_scheduler(flight_costs, raw_args, expected):
    params = parse_schedule_trip_json(raw_args)
    # TODO: don't need two separate params
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
    scheduler.create_model()
    scheduler.solve()
    result_records = scheduler.get_result_flight_records()
    # TODO better test cases
    assert len(result_records) == 5
    assert [r["day"] >= 1 and r["day"] <= 30 for r in result_records]
