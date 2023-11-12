import datetime
import requests
import threading
from queue import Queue
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
import os
import sys
import pickle
import fire
import pandas as pd


@retry(
    stop=stop_after_attempt(10),
    #wait=wait_random_exponential(multiplier=1, max=64)
    wait=wait_fixed(1)
)
def get_daily_flights_from(airport, max_calls=100,
                           lock=None,
                           airport_queue=None,
                           known_airports=None):
    offset = 0
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    dates = set([today])
    data = []
    limit = 100
    seen_flight_numbers = set()
    ncalls = 0
    while len(dates) == 1 and ncalls < max_calls:
        print("Getting from offset", offset)
        resp = requests.get('http://api.aviationstack.com/v1/flights', {'offset': offset, 'limit': limit, 'dep_iata': airport, 'access_key':
os.environ['AVIATIONSTACK_API_KEY']})
        resp.raise_for_status()
        ncalls += 1
        print(ncalls)
        sys.stdout.flush()
        dates = dates | set(d['flight_date'] for d in resp.json()['data'])
        new_data = []
        for d in resp.json()['data']:
            if d['flight']['number'] in seen_flight_numbers:
                continue
            if d['flight_date'] != today:
                continue
            seen_flight_numbers.add(d['flight']['number'])
            new_data.append(d)
        data.extend(new_data)
        if airport_queue is not None and known_airports is not None and lock is not None:
            for flight in new_data:
                if flight['arrival']['iata'] not in known_airports:
                    with lock:
                        known_airports.add(flight['arrival']['iata'])
                    airport_queue.put(flight['arrival']['iata'])

        count = resp.json()['pagination']['count']
        if count < limit:
            break
        offset += limit
    return data, ncalls

def get_daily_flights_crawl(start_airport='LAX', max_total_calls=100, output_file='flights.pickle'):
    prev_known_airports = 0
    known_airports = set([start_airport])
    queue = [start_airport]
    all_flights = []
    total_calls = 0
    while prev_known_airports < len(known_airports) and total_calls < max_total_calls:
        print("Known airport size:", len(known_airports))
        cur_airport = queue.pop(0)
        try:
            flights, ncalls = get_daily_flights_from(cur_airport, max_calls=max_total_calls - total_calls)
        except tenacity.RetryError:
            return all_flights
        with open(output_file, 'wb') as f:
            pickle.dump(all_flights, f)
        total_calls += ncalls
        all_flights.extend(flights)
        prev_known_airports = len(known_airports)
        new_airports = set(f['departure']['iata'] for f in flights)
        for airport in new_airports:
            if airport not in known_airports:
                queue.append(airport)
                known_airports.add(airport)
    return all_flights



def get_daily_flights_crawl_multithreaded(start_airport='LAX', max_total_calls=100, num_worker_threads=10, output_file='flights.pickle'):
    lock = threading.Lock()
    queue = Queue()
    threads = []
    known_airports = set([start_airport])
    all_flights = []
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
                        lock=lock)
                except tenacity.RetryError:
                    print("retry error")
                    break

                print("attempting to get lock")
                sys.stdout.flush()
                with lock:
                    sys.stdout.flush()
                    all_flights.extend(flights)
                    with open(output_file, 'wb') as f:
                        pickle.dump(all_flights, f)
                    new_airports = set(f['departure']['iata'] for f in flights)
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

    return all_flights


def main(max_total_calls: int = 1000, output_file: str = 'flights.pickle', multithreaded: bool = True):
    if multithreaded:
        flights = get_daily_flights_crawl_multithreaded(max_total_calls=max_total_calls, output_file=output_file)
    else:
        flights = get_daily_flights_crawl(max_total_calls=max_total_calls, output_file=output_file)
    print(len(flights), "flights")
    with open(output_file, 'wb') as f:
        pickle.dump(flights, f)


def build_flight_costs(flights):
    flight_costs = {}
    for flight in flights:
        departure, arrival = flight['departure']['iata'], flight['arrival']['iata']
        arrival_time = pd.Timestamp(flight['arrival']['scheduled'])
        departure_time = pd.Timestamp(flight['departure']['scheduled'])
        duration = arrival_time - departure_time
        if (departure, arrival) not in flight_costs:
            flight_costs[(departure, arrival)] = duration.total_seconds() / 3600
            flight_costs[(arrival, departure)] = duration.total_seconds() / 3600
    return flight_costs

def build_flight_costs_from_remote_file(bucket, filename):
    download_file_from_gcs(filename, filename, bucket)
    with open(filename, 'rb') as f:
        flights = pickle.load(f)
    flight_costs = build_flight_costs(flights)


if __name__ == '__main__':
    # fire.Fire(main)
    max_total_calls = 20
    flights = get_daily_flights_crawl_multithreaded(max_total_calls=max_total_calls)
    print(len(flights), "flights")
