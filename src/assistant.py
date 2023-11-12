from openai import OpenAI
from dotenv import load_dotenv
from tools import tools
import requests
import os
import json
FUNCTIONS_URL = os.getenv("FUNCTIONS_URL")

load_dotenv("../.env")

client = OpenAI()

#  intro_text = """
#  You are a a travel planner and booking agent, responsible for finding flights, lodging options, things to do, and all-in-one packages to fit the constraints and desires of the customer.
#  You should respond succinctly with specific details in a step-by-step manner that can immediately be acted upon, removing much of the paradox of choice from the customer.
#  """
intro_text = """
You are a a travel planner and booking agent, responsible for scheduling a trip and finding flights to fit the constraints and desires of the customer.
You should respond succinctly with specific details in a step-by-step manner that can immediately be acted upon, removing much of the paradox of choice from the customer.
Take advantage of the stateful tools to keep track of the customer's preferences and constraints,
and use the scheduling and flight tools to find the best flights for the customer.
"""



assistant = client.beta.assistants.create(
    name="Travel Planner",
    instructions=intro_text,
    tools=[{"type": "retrieval"}]+tools,
    model="gpt-4-1106-preview",
)

print("assistant id", assistant.id)

thread = client.beta.threads.create()
print("thread id", thread.id)

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="""
    I am a 30yo male living in Los Angeles, CA.
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

    My preferences are for short trip duration over low price, but stick to my budget and use economy class.
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
    if run.status == 'requires_action':
        print("action", run.required_action)
        action = run.required_action
        if action.type == "submit_tool_outputs":
            tool_calls = action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for call in tool_calls:
                print("schedule trip", call)
                response = requests.post(
                    FUNCTIONS_URL + f"/?function={call.function.name}",
                    json=json.loads(call.function.arguments)
                )
                response.raise_for_status()
                tool_outputs.append(
                    {'tool_call_id': call.id, 'output': json.dumps(response.json())}
                )

            run = client.beta.threads.runs.submit_tool_outputs(
              thread_id=thread.id,
              run_id=run.id,
              tool_outputs=tool_outputs,
            )
        else:
            raise Exception(f"unknown action type {action.type}")
    elif run.status == "completed":
        break

messages = client.beta.threads.messages.list(thread_id=thread.id)

# TODO needs to be robust to input arguments missing
