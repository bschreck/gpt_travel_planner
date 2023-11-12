from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../.env")

client = OpenAI()

intro_text = """
You are a a travel planner and booking agent, responsible for finding flights, lodging options, things to do, and all-in-one packages to fit the constraints and desires of the customer.
You should respond succinctly with specific details in a step-by-step manner that can immediately be acted upon, removing much of the paradox of choice from the customer.
"""

schedule_trip_prompt = """
You have a trip flight scheduler available to you via the "schedule_trip" function. It takes the following parameters:
- start_city: str (airport iata code, e.g. 'LAX')
- end_city: str (airport iata code, e.g. 'LAX')
- ndays: int, total number of days
- contiguous_sequence_constraints: A list of constraints on the duration of stay in each city. Each constraint is a dictionary with the following keys:
  - city: str (airport iata code, e.g. 'LAX')
  - hard_min: float, The minimum number of days to spend in the city.
  - soft_min: float, The minimum number of days to spend in the city, if possible.
  - soft_max: float, The maximum number of days to spend in the city, if possible.
  - hard_max: float, The maximum number of days to spend in the city.
  - max_visits: int, The maximum number of visits to the city.
- date_range_constraints: A list of constraints on the ordering of city visits on the trip. Each constraint is a dictionary with the following keys:
  - city: str (airport iata code, e.g. 'LAX')
  - min_start_day: int (between 1 and ndays)
  - max_start_day: int (between 1 and ndays)
  - min_end_day: int (between 1 and ndays)
  - max_end_day: int (between 1 and ndays)

The result is a list of dicts representing flights, each with:
- origin: str (airport iata code, e.g. 'LAX')
- destination: str (airport iata code, e.g. 'LAX')
- day: int, between 1 and ndays
"""

flight_picker_prompt = """
You have a flight picker available to you via the "flight_picker" function. It takes the following parameters:
- passenger name: str
- max_flights: int, the maximum number of flights to return
- flight_legs: A list of flight legs, each with:
  - origin: str (airport iata code, e.g. 'LAX')
  - destination: str (airport iata code, e.g. 'LAX')
  - earliest_departure_time: str (ISO 8601 format w/ time zone, e.g. '2024-01-07T08:00:00-08:00')
  - latest_arrival_time: str (ISO 8601 format w/ time zone, e.g. '2024-01-07T08:00:00-08:00')
The result is a list of dicts representing flights, each with:
# TODO

"""
passenger_prefs_prompt = """
"""

assistant = client.beta.assistants.create(
    name="Travel Planner",
    instructions=intro_text,
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview",
)
print("assistant id", assistant.id)

thread = client.beta.threads.create()
print("thread id", thread.id)

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="""
    I want to spend December in Mexico and/or Central America. It is currently November 7th.
    My goal is to see and explore historical/cultural places, to relax on the beach, to meet interesting people, and to get better at surfing.
    I want to spend part of the time at a surf camp with great surfing beaches for a beginner/intermediate surfer.
    I don't care about fancy lodging, but I do appreciate private rooms and bathrooms.
    I'll be traveling alone, and have a budget of $5000.
    Some other constraints:
    - I want to stay with my friends in Mexico City for a week before December 20th.
    - I don't care too much about Christmas, but it would be fun being somewhere with a lot of Christmas spirit.
    - I'm flying from LAX.
    - I want to spend 4 days in Baja (La Ventana) with my cousins who will be there Dec 15-31. They stay at Pelican Reef. I'd like to stay there or nearby.
    - I want to use either Aeromexico miles (via Amex transfer) or fly on United or American (since I have flight credits) as much as possible.
    """,
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Ben Schreck. The user has a premium account.",
)

print("run", run.id, run)
while True:
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run.status == "completed":
        break

messages = client.beta.threads.messages.list(thread_id=thread.id)
