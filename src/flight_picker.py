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

load_dotenv()


class TimeOfDay(enum.StrEnum):
    EARLY_MORNING = "early_morning"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    RED_EYE = "red_eye"

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
    time_of_day_order: dict[TimeOfDay, int]
    hard_max_cost: float
    soft_max_cost: float
    single_leg_hard_max_cost: float
    single_leg_soft_max_cost: float
    max_stops: int
    hard_max_duration: datetime.timedelta
    min_layover_duration: datetime.timedelta
    max_layover_duration: datetime.timedelta
    airline_preferences: dict[str, int]
    seat_class_prefernces: dict[SeatClass, int]
    seat_location_preference: SeatLocation
    seat_location_row_preference: SeatRow
    desires_extra_legroom: bool
    # TODO how to encode preferences for cost vs duration
    cost_sensitivity: float # must be tunable perhaps by asking the user questions

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
        return hash((
            self.origin,
            self.destination,
            self.earliest_departure_time,
            self.latest_arrival_time,
        ))


def persist_to_file(file_name):
    def decorator(original_func):
        try:
            cache = pickle.load(open(file_name, "rb"))
        except (IOError, ValueError):
            cache = {}

        def new_func(*args, **kwargs):
            key = args[0]
            if key not in cache:
                cache[key] = original_func(*args, **kwargs)
                pickle.dump(cache, open(file_name, "wb"))
            return cache[key]

        return new_func

    return decorator


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
        duffel_passengers = [
            {
                "type": "adult",
            }
        ]
    else:
        duffel_passengers = [p.to_duffel_dict() for p in passengers]


    @retry(
        wait=wait_random_exponential(multiplier=1, max=64),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ApiError, HTTPError)),
    )
    def reqfn():
        return (
            duffel.offer_requests.create()
            .cabin_class(str(cabin_class))
            .max_connections(max_connections)
            .passengers(duffel_passengers)
            .slices([
                flight.to_duffel_slice()
            ])
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
    '''
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
    '''
    if additional_passengers is None:
        additional_passengers = []
    ordered_possible_flights = []
    for i, leg in enumerate(flight_legs):
        offer_return = search_flights(
            leg,
            passengers=[user.passenger_info] + additional_passengers,
        )
        partial_offer_id = offer_return.id
        # have to show
        # slices[].segments[].operating_carrier.name
        ordered_possible_flights.append(order_and_remove_flights(
            offer_return.offers,
            user.flight_preferences,
            prev_legs=ordered_possible_flights[i-1] if i > 0 else None,
        ))
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
            time_of_day_order={
                TimeOfDay.EARLY_MORNING: 1,
                TimeOfDay.MORNING: 2,
                TimeOfDay.AFTERNOON: 3,
                TimeOfDay.EVENING: 4,
                TimeOfDay.RED_EYE: 5,
            },
            hard_max_cost=1000,
            soft_max_cost=500,
            single_leg_hard_max_cost=250,
            single_leg_soft_max_cost=200,
            max_stops=1,
            hard_max_duration=datetime.timedelta(hours=20),
            min_layover_duration=datetime.timedelta(hours=1),
            max_layover_duration=datetime.timedelta(hours=6),
            airline_preferences={
                "Aeromexico": 1,
                "United Airlines": 2,
                "American Airlines": 3,
            },
            seat_class_prefernces={
                SeatClass.ECONOMY: 1,
            },
            seat_location_preference=SeatLocation.AISLE,
            seat_location_row_preference=SeatRow.MIDDLE,
            desires_extra_legroom=True,
            cost_sensitivity=0.5,
        )
    )
    flight_legs = [
        DesiredFlightLeg(
            origin="LAX",
            destination="MEX",
            earliest_departure_time=pd.Timestamp('2023-12-01 07:00:00'),
            latest_arrival_time=pd.Timestamp('2023-12-02 7:00:00'),
        ),
        DesiredFlightLeg(
            origin="MEX",
            destination="OAX",
            earliest_departure_time=pd.Timestamp('2023-12-10 07:00:00'),
            latest_arrival_time=pd.Timestamp('2023-12-10 22:00:00'),
        ),
        DesiredFlightLeg(
            origin="OAX",
            destination="LAX",
            earliest_departure_time=pd.Timestamp('2023-12-20 07:00:00'),
            latest_arrival_time=pd.Timestamp('2023-12-21 7:00:00'),
        ),
    ]
    return user, flight_legs

def offer_total_duration(offer: Offer):
    segments = offer.slices[0].segments
    return segments[-1].arriving_at - segments[0].departing_at

def offer_is_desired_airline(offer: Offer, desired_airlines: dict[str, int]):
    return any(
        s.marketing_carrier.name in desired_airlines
        or s.operating_carrier.name in desired_airlines
        for s in offer.slices[0].segments
    )

class OfferMetrics:

    def __init__(self, compound_offer: list[Offer], flight_preferences: UserFlightPreferences):
        self.compound_offer = compound_offer
        self.flight_preferences = flight_preferences

    def sort_functions(self):
        return [
            lambda offers: sum(float(o.total_amount) for o in offers),
            # TODO is offer.total_duration available?
            lambda offers: max(offer_total_duration(o) for o in offers),
            lambda offers: sum(
                offer_is_desired_airline(o, self.flight_preferences.airline_preferences)
                for o in offers
            ) / len(offers),
        ]
    def __repr__(self):
        total_amount, total_duration, airline_preference_ratio = self.metrics
        return f"OfferMetrics(total_amount=${total_amount}, total_duration={total_duration}, airline_preference_ratio={airline_preference_ratio})"

    @property
    def metrics(self):
        return [fn(self.compound_offer) for fn in self.sort_functions()]


def order_and_remove_flights(
    offers: list[Offer],
    flight_preferences: UserFlightPreferences,
    prev_legs: list[list[Offer]] | None = None,
) -> list[tuple[list[Offer], OfferMetrics]]:
    possible_offers = []
    for offer in offers:
        if float(offer.total_amount) > flight_preferences.hard_max_cost:
            continue
        if len(offer.slices) > flight_preferences.max_stops + 1:
            continue
        assert len(offer.slices) == 1
        segments = offer.slices[0].segments
        total_duration = offer_total_duration(offer)
        if total_duration > flight_preferences.hard_max_duration:
            continue
        if len(segments) > 1:
            layover_durations = [
                segments[i+1].departing_at - segments[i].arriving_at
                for i in range(len(segments)-1)
            ]
            if any(d < flight_preferences.min_layover_duration for d in layover_durations):
                continue
            if any(d > flight_preferences.max_layover_duration for d in layover_durations):
                continue
        possible_offers.append(offer)
    # TODO actually run a constraint optimizer
    # and for now, sort by some weighted combination of cost, duration, and airline preference
    # instead of just one by one
    if prev_legs:
        compound_offers = []
        for prev_compound_offer in prev_legs:
            for offer in possible_offers:
                compound_offers.append([*prev_compound_offer[0], offer])
    else:
        compound_offers = [[offer] for offer in possible_offers]
    return sorted([
        [co, OfferMetrics(co, flight_preferences)]
        for co in compound_offers
    ], key=lambda x: x[1].metrics)


if __name__ == "__main__":
    user, flight_legs = seed_data()
    compound_offers = find_and_select_flights(
        user=user,
        flight_legs=flight_legs,
    )
    print(len(compound_offers))
    for i, (compound_offer, metrics) in enumerate(compound_offers):
        print(f"Compound Offer {i}: ${metrics}")
        for j, offer in enumerate(compound_offer):
            print(f"Offer {j}: ${offer.total_amount}")
            for slice in offer.slices:
                print(f"{slice.origin.iata_city_code} -> {slice.destination.iata_city_code}")
                print(f"Segments: {len(slice.segments)}")
                for segment in slice.segments:
                    print(f"{segment.operating_carrier.name}, marketed by {segment.marketing_carrier.name}")
                    print(f"{segment.origin.iata_code} {segment.departing_at} -> {segment.destination.iata_code} {segment.arriving_at}")
