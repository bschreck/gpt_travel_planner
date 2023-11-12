import os
import pandas as pd
import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from duffel_api import Duffel
from duffel_api.models import Offer
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random_exponential,
    stop_after_attempt,
)
from duffel_api.http_client import ApiError
from requests import HTTPError
import enum
from collections import OrderedDict
import pickle
import numpy as np

load_dotenv()


class TimeOfDay(enum.StrEnum):
    EARLY_MORNING = "early_morning"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    RED_EYE = "red_eye"

    @classmethod
    def map_time(cls, depart_time: datetime.time, arrive_time: datetime.time):
        hour = depart_time.hour
        if arrive_time < depart_time:
            return cls.RED_EYE
        elif hour <= 6:
            return cls.EARLY_MORNING
        elif hour <= 12:
            return cls.MORNING
        elif hour <= 17:
            return cls.AFTERNOON
        else:
            return cls.EVENING


class SeatClass(enum.StrEnum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class SeatRow(enum.StrEnum):
    FRONT = "front"
    MIDDLE = "middle"
    BACK = "back"


class SeatLocation(enum.StrEnum):
    AISLE = "aisle"
    WINDOW = "window"
    MIDDLE = "middle"


@dataclass
class UserFlightPreferences:
    time_of_day_order: list[TimeOfDay]
    hard_max_cost: float
    soft_max_cost: float
    single_leg_hard_max_cost: float
    single_leg_soft_max_cost: float
    soft_max_duration: datetime.timedelta
    hard_max_duration: datetime.timedelta

    soft_max_stops: int
    hard_max_stops: int
    soft_min_layover_duration: datetime.timedelta
    hard_min_layover_duration: datetime.timedelta
    soft_max_layover_duration: datetime.timedelta
    hard_max_layover_duration: datetime.timedelta

    airline_preferences: dict[str, int]
    seat_class_prefernces: dict[SeatClass, int]
    seat_location_preference: SeatLocation
    seat_location_row_preference: SeatRow
    desires_extra_legroom: bool
    # TODO how to encode preferences for cost vs duration
    # cost_sensitivity: float # must be tunable perhaps by asking the user questions
    total_cost_weight: float = 0.4
    total_duration_weight: float = 0.2
    preferred_airline_ratio_weight: float = 0.05
    time_of_day_weight: float = 0.15
    layover_duration_weight: float = 0.1
    nstops_weight: float = 0.1

    def __post_init__(self):
        assert np.isclose(
            self.total_cost_weight
            + self.total_duration_weight
            + self.preferred_airline_ratio_weight
            + self.time_of_day_weight
            + self.layover_duration_weight
            + self.nstops_weight,
            1,
        )


class PassengerType(enum.StrEnum):
    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"
    SENIOR = "senior"
    STUDENT = "student"
    YOUTH = "youth"


@dataclass
class PassengerInfo:
    name: str
    date_of_birth: datetime.date
    phone_number: str
    email: str
    age: int | None = None
    ptype: PassengerType = PassengerType.ADULT

    def to_duffel_dict(self):
        d = {
            "family_name": self.name.rsplit(" ", 1)[1],
            "given_name": self.name.rsplit(" ", 1)[0],
            "date_of_birth": self.date_of_birth.strftime("%Y-%m-%d"),
        }
        if self.age:
            d["age"] = self.age
        else:
            d["type"] = self(self.ptype)
        return d


@dataclass
class User:
    email: str
    passenger_info: PassengerInfo
    flight_preferences: UserFlightPreferences


@dataclass
class DesiredFlightLeg:
    origin: str
    destination: str
    earliest_departure_time: datetime.datetime
    latest_arrival_time: datetime.datetime

    def to_duffel_slice(self):
        return {
            "origin": self.origin,
            "destination": self.destination,
            "departure_date": self.earliest_departure_time.strftime("%Y-%m-%d"),
        }

    def __hash__(self):
        return hash(
            (
                self.origin,
                self.destination,
                self.earliest_departure_time,
                self.latest_arrival_time,
            )
        )


def persist_to_file(file_name):
    def decorator(original_func):
        try:
            cache = pickle.load(open(file_name, "rb"))
        except (IOError, ValueError, AttributeError):
            cache = {}

        def new_func(*args, **kwargs):
            key = args[0]
            if key not in cache:
                cache[key] = original_func(*args, **kwargs)
                pickle.dump(cache, open(file_name, "wb"))
            return cache[key]

        return new_func

    return decorator


# TODO lru_cache with timeout
@persist_to_file("flight_picker_cache.pickle")
def search_flights(
    flight: DesiredFlightLeg,
    max_connections=1,
    cabin_class: SeatClass = SeatClass.ECONOMY,
    passengers: list[PassengerInfo] | None = None,
):
    # TODO partial offers?
    # TODO can combine slices/legs of different offers
    duffel = Duffel(access_token=os.environ["DUFFEL_API_KEY"])

    if passengers is None:
        duffel_passengers = [{"type": "adult"}]
    else:
        duffel_passengers = [p.to_duffel_dict() for p in passengers]

    return (
        duffel.offer_requests.create()
        .cabin_class(str(cabin_class))
        .max_connections(max_connections)
        .passengers(duffel_passengers)
        .slices([flight.to_duffel_slice()])
        .return_offers()
        .execute()
    )

    @retry(
        wait=wait_random_exponential(multiplier=1, max=64),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(HTTPError),
        # TODO retry on specific ApiErrors
    )
    def reqfn():
        return (
            duffel.offer_requests.create()
            .cabin_class(str(cabin_class))
            .max_connections(max_connections)
            .passengers(duffel_passengers)
            .slices([flight.to_duffel_slice()])
            .return_offers()
            .execute()
        )

    # TODO can make async by not using .return_offers() and then calling
    # offers = client.offers.list(offer_request.id)
    return reqfn()


def find_and_select_flights(
    user: User,
    flight_legs: list[DesiredFlightLeg],
    additional_passengers: list[PassengerInfo] | None = None,
    max_flights_to_return=10,
):
    """
    Constraints/prefs:
    - desired time of day
    - required departure or arrival time
    - cost
    - Stops
    - total duration
    - layover duration
    - airline
    - seat class
    - specific seat prefs
    """
    if additional_passengers is None:
        additional_passengers = []
    ordered_possible_flights = []
    for i, leg in enumerate(flight_legs):
        print("searching leg", i)
        offer_return = search_flights(
            leg, passengers=[user.passenger_info] + additional_passengers
        )
        print(f"found {len(offer_return.offers)} offers")
        partial_offer_id = offer_return.id
        # have to show
        # slices[].segments[].operating_carrier.name
        ordered_possible_flights.append(
            order_and_remove_flights(
                offer_return.offers,
                user.flight_preferences,
                prev_legs=ordered_possible_flights[i - 1] if i > 0 else None,
            )
        )
    return ordered_possible_flights[-1][:max_flights_to_return]


def book_flight(flight_details):
    """
    create payment intent with duffel
    In frontend, use duffel payment component to collect payment details (see if I can get around this)
    - maybe GPT provides a link to a frontend site with the payment component, and says to return here when done
    Then confirm a payment intent with duffel
    Then create an order with duffel
    """
    pass


def seed_data():
    user = User(
        email="ben@schreck.com",
        passenger_info=PassengerInfo(
            name="Ben Schreck",
            date_of_birth=datetime.date(1993, 1, 1),
            phone_number="555-555-5555",
            email="ben@schreck.com",
            age=30,
        ),
        flight_preferences=UserFlightPreferences(
            time_of_day_order=[
                TimeOfDay.MORNING,
                TimeOfDay.AFTERNOON,
                TimeOfDay.EVENING,
                TimeOfDay.EARLY_MORNING,
                TimeOfDay.RED_EYE,
            ],
            hard_max_cost=1000,
            soft_max_cost=500,
            single_leg_hard_max_cost=250,
            single_leg_soft_max_cost=200,
            soft_max_stops=1,
            hard_max_stops=2,
            soft_min_layover_duration=datetime.timedelta(hours=1.5),
            hard_min_layover_duration=datetime.timedelta(hours=0.75),
            soft_max_layover_duration=datetime.timedelta(hours=2),
            hard_max_layover_duration=datetime.timedelta(hours=6),
            soft_max_duration=datetime.timedelta(hours=5),
            hard_max_duration=datetime.timedelta(hours=20),
            airline_preferences={
                "Aeromexico": 1,
                "United Airlines": 2,
                "American Airlines": 3,
            },
            seat_class_prefernces={SeatClass.ECONOMY: 1},
            seat_location_preference=SeatLocation.AISLE,
            seat_location_row_preference=SeatRow.MIDDLE,
            desires_extra_legroom=True,
            #  total_cost_weight = 0.4,
            #  total_duration_weight= 0.2,
            #  preferred_airline_ratio_weight= 0.05,
            #  time_of_day_weight = 0.15,
            #  layover_duration_weight= 0.1,
            #  nstops_weight= 0.1,
            total_cost_weight=0.1,
            total_duration_weight=0.3,
            preferred_airline_ratio_weight=0.5,
            time_of_day_weight=0.02,
            layover_duration_weight=0.04,
            nstops_weight=0.04,
        ),
    )
    flight_legs = [
        DesiredFlightLeg(
            origin="LAX",
            destination="MEX",
            earliest_departure_time=pd.Timestamp("2023-12-01 07:00:00"),
            latest_arrival_time=pd.Timestamp("2023-12-02 7:00:00"),
        ),
        DesiredFlightLeg(
            origin="MEX",
            destination="OAX",
            earliest_departure_time=pd.Timestamp("2023-12-10 07:00:00"),
            latest_arrival_time=pd.Timestamp("2023-12-10 22:00:00"),
        ),
        DesiredFlightLeg(
            origin="OAX",
            destination="LAX",
            earliest_departure_time=pd.Timestamp("2023-12-20 07:00:00"),
            latest_arrival_time=pd.Timestamp("2023-12-21 7:00:00"),
        ),
    ]
    return user, flight_legs


def nonlinear_cost_mapping(cost, soft_max, hard_max, decay_start_value):
    """
    Maps a cost value to a nonlinear output with two phases: linear and exponential decay.

    :param cost: float, the cost value to map, analogous to the duration in seconds
    :param soft_max: float, the threshold at which the output starts to decrease exponentially, analogous to softmax_duration in seconds
    :param hard_max: float, the threshold at which the output becomes 0, analogous to hard_max_duration in seconds
    :param decay_start_value: float, the value at the start of the exponential decay phase, analogous to the y-cutoff value
    :return: float, the mapped value
    """
    # Check if the cost is beyond the hard max threshold
    if cost >= hard_max:
        return 0

    # Linearly decreasing phase
    if cost <= soft_max:
        # Linear interpolation between (0, 1) and (soft_max, decay_start_value)
        return linear_interpolation(cost, 0, soft_max, 1, decay_start_value)
    else:
        # Exponentially decreasing phase
        # Calculate the exponent based on the remaining threshold
        remaining_threshold = hard_max - soft_max
        remaining_cost = cost - soft_max

        # Exponential function: decay_start_value * e^(-x) where x is scaled to the remaining threshold
        scale = -np.log(decay_start_value) / remaining_threshold
        return decay_start_value * np.exp(-scale * remaining_cost)


def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)


# TODO this is wrong or more variables can be added like steepness
def exp_interpolation(x, x1, x2, y1, y2):
    scale = np.log(y2 / y1) / (x2 - x1)
    return y1 / y2 * np.exp(scale * (x - x1))


# TODO this is wrong
def concave_down_exp(x, x1, x2, y1, y2, steepness=1):
    b = np.log(y1 + steepness - y2) / (x2 - x1)
    return (y1 + steepness) - steepness * np.exp(b * (x - x1))


def inverse_nonlinear_cost_mapping_4_phase(
    cost,
    soft_min,
    hard_min,
    soft_max,
    hard_max,
    decay_1_start_value=0.25,
    decay_2_start_value=0.75,
):
    if cost <= hard_min:
        return 1
    elif cost >= hard_max:
        return 1
    elif cost <= soft_min:
        return concave_down_exp(cost, hard_min, soft_min, 1, decay_1_start_value)
        ## exponentially decreasing phase (hard_min, 1) and (soft_min, decay_1_start_value)
        # remaining_threshold = soft_min - hard_min
        # remaining_cost = soft_min - cost
        # scale = -np.log(1/decay_1_start_value) / remaining_threshold
        # return decay_1_start_value * np.exp(-scale * remaining_cost)
    elif cost >= soft_min and cost <= soft_max:
        # Linear interpolation between (soft_min, decay_1_start_value), (soft_max, decay_2_start_value)
        return linear_interpolation(
            cost, soft_min, soft_max, decay_1_start_value, decay_2_start_value
        )
    else:
        return exp_interpolation(cost, soft_max, hard_max, decay_2_start_value, 1)
        ## Exponentially increasing phase between (soft_max, decay_2_start_value) and (hard_max, 1)
        # remaining_threshold = hard_max - soft_max
        # remaining_cost = cost - soft_max
        # scale = np.log(1/decay_2_start_value) / remaining_threshold
        # return decay_2_start_value * np.exp(scale * remaining_cost)


def offers_total_cost_score(
    offers: list[Offer],
    soft_max_cost: float,
    hard_max_cost: float,
    decay_start_value: float = 0.75,
):
    total_cost = sum(float(o.total_amount) for o in offers)
    return nonlinear_cost_mapping(
        total_cost, soft_max_cost, hard_max_cost, decay_start_value
    )


def offer_total_duration(offer: Offer):
    segments = offer.slices[0].segments
    return segments[-1].arriving_at - segments[0].departing_at


def offer_total_duration_score(
    offer: Offer,
    soft_max_duration: datetime.timedelta,
    hard_max_duration: datetime.timedelta,
    decay_start_value: float = 0.75,
):
    soft_max_seconds = soft_max_duration.total_seconds()
    hard_max_seconds = hard_max_duration.total_seconds()
    total_duration = offer_total_duration(offer).total_seconds()
    return nonlinear_cost_mapping(
        total_duration, soft_max_seconds, hard_max_seconds, decay_start_value
    )


def offers_total_duration_score(
    offers: Offer,
    soft_max_duration: datetime.timedelta,
    hard_max_duration: datetime.timedelta,
    decay_start_value: float = 0.75,
):
    return max(
        offer_total_duration_score(
            o, soft_max_duration, hard_max_duration, decay_start_value
        )
        for o in offers
    )


# TODO should use order of desired airlines instead of binary
def offer_is_desired_airline(offer: Offer, desired_airlines: dict[str, int]):
    return any(
        s.marketing_carrier.name in desired_airlines
        or s.operating_carrier.name in desired_airlines
        for s in offer.slices[0].segments
    )


def offer_time_of_days(offers: list[Offer]):
    return [
        TimeOfDay.map_time(s.departing_at.time(), s.arriving_at.time())
        for o in offers
        for s in o.slices[0].segments
    ]


def offer_time_of_day_score(offers: list[Offer], time_of_day_order: list[TimeOfDay]):
    segments = [s for o in offers for s in o.slices[0].segments]

    weights_from_order = OrderedDict(
        {
            time_of_day: len(time_of_day_order) / (i + 1)
            for i, time_of_day in enumerate(time_of_day_order)
        }
    )
    scores = []
    for s in segments:
        depart_time = s.departing_at.time()
        arrive_time = s.arriving_at.time()
        time_of_day = TimeOfDay.map_time(depart_time, arrive_time)
        scores.append(weights_from_order[time_of_day])
    return np.average(scores) / len(time_of_day_order)


def offers_nstops_score(offers, soft_max_stops, hard_max_stops, decay_start_value=0.75):
    return nonlinear_cost_mapping(
        sum(len(o.slices[0].segments) - 1 for o in offers),
        soft_max_stops,
        hard_max_stops,
        decay_start_value,
    )


def offers_layover_duration_score(
    offers,
    hard_min_layover_duration,
    soft_min_layover_duration,
    soft_max_layover_duration,
    hard_max_layover_duration,
):
    layover_duration_scores = []
    for o in offers:
        if len(o.slices[0].segments) > 1:
            layover_durations = [
                (s.arriving_at - s.departing_at).total_seconds()
                for s in o.slices[0].segments[:-1]
            ]
            layover_duration_scores.extend(
                [
                    inverse_nonlinear_cost_mapping_4_phase(
                        ld,
                        soft_min_layover_duration.total_seconds(),
                        hard_min_layover_duration.total_seconds(),
                        soft_max_layover_duration.total_seconds(),
                        hard_max_layover_duration.total_seconds(),
                    )
                    for ld in layover_durations
                ]
            )
        else:
            layover_duration_scores.append(0)
    # TODO does this need to be scaled to 0,1?
    return np.average(layover_duration_scores)


def offers_layover_durations(offers):
    layover_durations = []
    for o in offers:
        if len(o.slices[0].segments) > 1:
            layover_durations.extend(
                [
                    (s.arriving_at - s.departing_at).total_seconds() / 3600
                    for s in o.slices[0].segments[:-1]
                ]
            )
        else:
            layover_durations.append(0)
    return layover_durations


def offers_nstops(offers):
    return sum(len(o.slices[0].segments) - 1 for o in offers)


class OfferMetrics:
    def __init__(
        self, compound_offer: list[Offer], flight_preferences: UserFlightPreferences
    ):
        self.compound_offer = compound_offer
        self.flight_preferences = flight_preferences

    @property
    def display_metric_functions(self):
        return {
            "total_cost": lambda offers: sum(float(o.total_amount) for o in offers),
            # TODO is offer.total_duration available?
            "max_total_duration": lambda offers: max(
                offer_total_duration(o).total_seconds() / 3600 for o in offers
            ),
            "time_of_days": lambda offers: [
                str(td) for td in offer_time_of_days(offers)
            ],
            "is_preferred_airline": lambda offers: [
                offer_is_desired_airline(o, self.flight_preferences.airline_preferences)
                for o in offers
            ],
            "layover_durations": offers_layover_durations,
            "nstops": offers_nstops,
        }

    @property
    def sort_functions(self):
        return {
            "total_cost": lambda offers: offers_total_cost_score(
                offers,
                self.flight_preferences.soft_max_cost,
                self.flight_preferences.hard_max_cost,
            ),
            # TODO is offer.total_duration available?
            "total_duration": lambda offers: offers_total_duration_score(
                offers,
                self.flight_preferences.soft_max_duration,
                self.flight_preferences.hard_max_duration,
            ),
            "time_of_day": lambda offers: offer_time_of_day_score(
                offers, self.flight_preferences.time_of_day_order
            ),
            "preferred_airline_ratio": lambda offers: sum(
                offer_is_desired_airline(o, self.flight_preferences.airline_preferences)
                for o in offers
            )
            / len(offers),
            "layover_duration": lambda offers: offers_layover_duration_score(
                offers,
                self.flight_preferences.hard_min_layover_duration,
                self.flight_preferences.soft_min_layover_duration,
                self.flight_preferences.soft_max_layover_duration,
                self.flight_preferences.hard_max_layover_duration,
            ),
            "nstops": lambda offers: offers_nstops_score(
                offers,
                self.flight_preferences.soft_max_stops,
                self.flight_preferences.hard_max_stops,
            ),
        }

    def __repr__(self):
        display_metrics = "".join(
            ["\n    " + f"{key}={value}" for key, value in self.display_metrics.items()]
        )
        return f"OfferMetrics({display_metrics})"

    @property
    def display_metrics(self):
        return {
            name: fn(self.compound_offer)
            for name, fn in self.display_metric_functions.items()
        }

    @property
    def metrics(self):
        return {
            name: fn(self.compound_offer) for name, fn in self.sort_functions.items()
        }

    @property
    def weights(self):
        return np.array(
            [
                getattr(self.flight_preferences, f"{key}_weight")
                for key in self.sort_functions
            ]
        )

    @property
    def weighted_metrics(self):
        return self.weights * np.array(
            [fn(self.compound_offer) for fn in self.sort_functions.values()]
        )

    @property
    def weighted_metrics_sum(self):
        return sum(self.weighted_metrics)


def order_and_remove_flights(
    offers: list[Offer],
    flight_preferences: UserFlightPreferences,
    prev_legs: list[list[Offer]] | None = None,
) -> list[tuple[list[Offer], OfferMetrics]]:
    possible_offers = []
    for offer in offers:
        if float(offer.total_amount) > flight_preferences.hard_max_cost:
            continue
        if len(offer.slices) > flight_preferences.hard_max_stops + 1:
            continue
        assert len(offer.slices) == 1
        segments = offer.slices[0].segments
        total_duration = offer_total_duration(offer)
        if total_duration > flight_preferences.hard_max_duration:
            continue
        if len(segments) > 1:
            layover_durations = [
                segments[i + 1].departing_at - segments[i].arriving_at
                for i in range(len(segments) - 1)
            ]
            if any(
                d < flight_preferences.hard_min_layover_duration
                for d in layover_durations
            ):
                continue
            if any(
                d > flight_preferences.hard_max_layover_duration
                for d in layover_durations
            ):
                continue
        possible_offers.append(offer)
    # TODO actually run a constraint optimizer
    # and for now, sort by some weighted combination of flight preference scores
    # instead of just one by one
    if prev_legs:
        compound_offers = []
        for prev_compound_offer in prev_legs:
            for offer in possible_offers:
                compound_offers.append([*prev_compound_offer[0], offer])
    else:
        compound_offers = [[offer] for offer in possible_offers]
    return sorted(
        [[co, OfferMetrics(co, flight_preferences)] for co in compound_offers],
        key=lambda x: -x[1].weighted_metrics_sum,
    )


def offer_to_json(offer: Offer) -> dict:
    return {
        "total_amount": float(offer.total_amount),
        "slices": [
            {
                "segments": [
                    {
                        "origin": segment.origin.iata_code,
                        "destination": segment.destination.iata_code,
                        "departing_at": segment.departing_at.isoformat(),
                        "arriving_at": segment.arriving_at.isoformat(),
                        "operating_carrier": segment.operating_carrier.name,
                        "marketing_carrier": segment.marketing_carrier.name,
                        "duration": (
                            segment.arriving_at - segment.departing_at
                        ).total_seconds(),
                    }
                    for segment in slice.segments
                ]
            }
            for slice in offer.slices
        ],
    }


def compound_offers_with_metrics_to_json(compound_offers_with_metrics) -> list[dict]:
    compound_offers_as_dicts = []
    for compound_offer, metrics in compound_offers_with_metrics:
        compound_offers_as_dicts.append(
            {
                "metrics": metrics.metrics,
                "display_metrics": metrics.display_metrics,
                "offers": [offer_to_json(o) for o in compound_offer],
            }
        )
    return compound_offers_as_dicts


if __name__ == "__main__":
    user, flight_legs = seed_data()
    compound_offers = find_and_select_flights(
        user=user, flight_legs=flight_legs, max_flights_to_return=10
    )
    compound_offers_as_dicts = compound_offers_with_metrics_to_json(compound_offers)
    print(compound_offers_as_dicts[0])
    #  for i, (compound_offer, metrics) in enumerate(compound_offers):
    #  print(f"Compound Offer {i}: {metrics}")
    #  for j, offer in enumerate(compound_offer):
    #  print(offer.to_json())
    #  print(f"Offer {j}: ${offer.total_amount}")
    #  for slice in offer.slices:
    #  print(f"{slice.origin.iata_city_code} -> {slice.destination.iata_city_code}")
    #  print(f"Segments: {len(slice.segments)}")
    #  for segment in slice.segments:
    #  print(f"{segment.operating_carrier.name}, marketed by {segment.marketing_carrier.name}")
    #  print(f"{segment.origin.iata_code} {segment.departing_at} -> {segment.destination.iata_code} {segment.arriving_at}")
