import datetime
import requests
import threading
from queue import Queue
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed,
    RetryError,
)
import os
import sys
import pickle
import fire
import pandas as pd
from utils import (
    upload_file_to_gcs,
    download_file_from_gcs,
    cache_with_ttl,
    persist_to_file,
)
from config import (
    DEFAULT_IATA_CODES_FILE,
    DEFAULT_FLIGHT_COSTS_FILE,
    DEFAULT_FLIGHTS_FILE
)


def open_and_save_new_data(new_data, output_file, bucket):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, "rb") as f:
            existing_data = pickle.load(f)
    except (IOError, ValueError):
        existing_data = []
    with open(output_file, "wb") as f:
        pickle.dump(new_data + existing_data, f)
    if bucket is not None:
        upload_file_to_gcs(output_file, output_file, bucket)


# @retry(
#    stop=stop_after_attempt(10),
#    #wait=wait_random_exponential(multiplier=1, max=64)
#    wait=wait_fixed(1)
# )
def get_daily_flights_from(
    airport,
    max_calls=100,
    lock=None,
    airport_queue=None,
    known_airports=None,
    output_file="flights.pickle",
    bucket=None,
):
    offset = 0
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    data = []
    limit = 100
    seen_flight_numbers = set()
    ncalls = 0
    while ncalls < max_calls:
        resp = requests.get(
            "http://api.aviationstack.com/v1/routes",
            {
                "offset": offset,
                "limit": limit,
                "dep_iata": airport,
                "access_key": os.environ["AVIATIONSTACK_API_KEY"],
            },
        )
        resp.raise_for_status()
        ncalls += 1
        sys.stdout.flush()
        new_data = []
        for d in resp.json()["data"]:
            if d["flight"]["number"] in seen_flight_numbers:
                continue
            seen_flight_numbers.add(d["flight"]["number"])
            new_data.append(d)
        data.extend(new_data)
        if (
            airport_queue is not None
            and known_airports is not None
            and lock is not None
        ):
            for flight in new_data:
                if flight["arrival"]["iata"] not in known_airports:
                    with lock:
                        known_airports.add(flight["arrival"]["iata"])
                    airport_queue.put(flight["arrival"]["iata"])

        if lock is not None:
            with lock:
                open_and_save_new_data(data, output_file, bucket)

        count = resp.json()["pagination"]["count"]
        if count < limit:
            break
        offset += limit
    return data, ncalls


def get_daily_flights_crawl(
    start_airport="LAX", max_total_calls=100, output_file="flights.pickle"
):
    prev_known_airports = 0
    known_airports = set([start_airport])
    queue = [start_airport]
    all_flights = []
    total_calls = 0
    while prev_known_airports < len(known_airports) and total_calls < max_total_calls:
        cur_airport = queue.pop(0)
        try:
            flights, ncalls = get_daily_flights_from(
                cur_airport, max_calls=max_total_calls - total_calls
            )
        except RetryError:
            return all_flights
        open_and_save_new_data(all_flights, output_file, bucket)
        total_calls += ncalls
        all_flights.extend(flights)
        prev_known_airports = len(known_airports)
        new_airports = set(f["departure"]["iata"] for f in flights)
        for airport in new_airports:
            if airport not in known_airports:
                queue.append(airport)
                known_airports.add(airport)
    return all_flights


def get_daily_flights_crawl_multithreaded(
    start_airport="LAX",
    max_total_calls=100,
    num_worker_threads=10,
    output_file="flights.pickle",
    bucket=None,
):
    lock = threading.Lock()
    queue = Queue()
    threads = []
    known_airports = set([start_airport])
    total_calls = [0]

    def worker():
        try:
            while True:
                sys.stdout.flush()
                cur_airport = queue.get()
                if cur_airport is None:
                    break
                if total_calls[0] >= max_total_calls:
                    break
                try:
                    flights, ncalls = get_daily_flights_from(
                        cur_airport,
                        max_calls=max_total_calls - total_calls[0],
                        airport_queue=queue,
                        known_airports=known_airports,
                        lock=lock,
                        output_file=output_file,
                        bucket=bucket,
                    )
                except RetryError:
                    print("retry error")
                    break

                sys.stdout.flush()
                with lock:
                    sys.stdout.flush()
                    open_and_save_new_data(flights, output_file, bucket)
                    new_airports = set(f["departure"]["iata"] for f in flights)
                    total_calls[0] += ncalls
                    for airport in new_airports:
                        if airport not in known_airports:
                            queue.put(airport)
                            known_airports.add(airport)
        except Exception as e:
            print(e)
            return
        queue.task_done()

    # Put the start airport in the queue
    queue.put(start_airport)

    # Start worker threads
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # Block until all tasks are done
    queue.join()

    # Stop workers
    for i in range(num_worker_threads):
        queue.put(None)
    for t in threads:
        t.join()


def main(
    max_total_calls: int = 1000,
    output_file_basename: str = "flights.pickle",
    multithreaded: bool = True,
    bucket: str = None,
):
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    output_file = f"{today}/{output_file_basename}"
    if multithreaded:
        get_daily_flights_crawl_multithreaded(
            max_total_calls=max_total_calls, output_file=output_file, bucket=bucket
        )
    else:
        get_daily_flights_crawl(
            max_total_calls=max_total_calls, output_file=output_file, bucket=bucket
        )


def build_flight_costs(flights):
    # TODO: this is slow
    flight_costs = {}
    for flight in flights:
        departure, arrival = flight["departure"]["iata"], flight["arrival"]["iata"]
        arrival_time = pd.Timestamp(flight["arrival"]["time"])
        departure_time = pd.Timestamp(flight["departure"]["time"])
        duration = arrival_time - departure_time
        if duration < pd.Timedelta(0):
            arrival_time += pd.Timedelta(1, unit="d")
            duration = arrival_time - departure_time
        if (departure, arrival) not in flight_costs:
            flight_costs[(departure, arrival)] = duration.total_seconds() / 3600
            flight_costs[(arrival, departure)] = duration.total_seconds() / 3600
    return flight_costs


# TODO ttl
@cache_with_ttl(ttl=60 * 60 * 24)
@persist_to_file("flights_cache.pickle")
def build_flight_costs_from_remote_file(bucket, remote_filename, local_filename):
    download_file_from_gcs(remote_filename, local_filename, bucket)
    with open(local_filename, "rb") as f:
        flights = pickle.load(f)
    return build_flight_costs(flights)


@cache_with_ttl(ttl=60 * 60 * 24)
@persist_to_file(DEFAULT_FLIGHT_COSTS_FILE)
def make_local_flight_costs_full(flights_local_file, flight_costs_local_file):
    from scheduler import get_approx_flight_data
    with open(flights_local_file, "rb") as f:
        flights = pickle.load(f)
    flight_costs = build_flight_costs(flights)
    flight_costs = get_approx_flight_data(flight_costs)
    with open(flight_costs_local_file, "wb") as f:
        pickle.dump(flight_costs, f)
    return flight_costs


@cache_with_ttl(ttl=60 * 60 * 24)
def iata_codes_from_file(filename=DEFAULT_IATA_CODES_FILE):
    return pd.read_csv(filename)

@cache_with_ttl(ttl=60 * 60 * 24)
def _get_iata_codes_by_country(country):
    flight_costs = make_local_flight_costs_full(DEFAULT_FLIGHTS_FILE, DEFAULT_FLIGHT_COSTS_FILE)
    available_airports = set()
    for origin, destination in flight_costs:
        available_airports.add(origin)
        available_airports.add(destination)
    iata_codes = iata_codes_from_file()
    relevant = iata_codes[
        (iata_codes['iso_country'] == country)
        & (iata_codes['type'].isin(['large_airport', 'medium_airport']))
    ]
    return {
        v[0]: v[1] for v in relevant[['iata_code', 'municipality']].values
        if isinstance(v[0], str) and isinstance(v[1], str)
        and v[0] in available_airports
    }

# TODO cleanup
@cache_with_ttl(ttl=60 * 60 * 24)
def get_iata_codes_by_country(request):
    country = request.get_json(silent=True)['country']
    return _get_iata_codes_by_country(country)

if __name__ == "__main__":
    fire.Fire(main)
