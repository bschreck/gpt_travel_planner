from openai import OpenAI
from dotenv import load_dotenv
from tools import tools
import requests
from requests.exceptions import JSONDecodeError
import os
import json
from config import get_env
import sys

ENV = get_env()
if ENV == "local":
    FUNCTIONS_URL = os.getenv("LOCAL_FUNCTIONS_URL")
else:
    FUNCTIONS_URL = os.getenv("FUNCTIONS_URL")

load_dotenv("../.env")

client = OpenAI()

def create_or_retrieve_assistant():
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


    assistant_id = os.getenv("ASSISTANT_ID")
    if not assistant_id:
        assistant = client.beta.assistants.create(
            name="Travel Planner",
            instructions=intro_text,
            tools=[{"type": "retrieval"}] + tools,
            model="gpt-4-1106-preview",
        )
    else:
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        print("retrieved assistant")

    print("assistant id", assistant.id)
    return assistant

def create_or_retrieve_user_thread(user_id):
    thread_id = os.getenv("THREAD_ID")
    thread_id = None
    if not thread_id:
        thread = client.beta.threads.create()
    else:
        thread = client.beta.threads.retrieve(thread_id=thread_id)
        print("retrieved thread")

    print("thread id", thread.id)
    return thread

#message = client.beta.threads.messages.create(
#    thread_id=thread.id,
#    role="user",
#    content="""
#    I am a 30yo male living in Los Angeles, CA.
#    I want to spend December in Mexico and/or Central America. It is currently November 7th.
#    My goal is to see and explore historical/cultural places, to relax on the beach, to meet interesting people, and to get better at surfing.
#    I want to spend part of the time at a surf camp with great surfing beaches for a beginner/intermediate surfer.
#    I don't care about fancy lodging, but I do appreciate private rooms and bathrooms.
#    I'll be traveling alone, and have a budget of $5000.
#    Some other constraints:
#    - I want to stay with my friends in Mexico City for a week before December 20th.
#    - I don't care too much about Christmas, but it would be fun being somewhere with a lot of Christmas spirit.
#    - I'm flying from LAX.
#    - I want to spend 4 days in Baja (La Ventana) with my cousins who will be there Dec 15-31. They stay at Pelican Reef. I'd like to stay there or nearby.
#    - I want to use either Aeromexico miles (via Amex transfer) or fly on United or American (since I have flight credits) as much as possible.
#
#    My preferences are for short trip duration over low price, but stick to my budget and use economy class.
#    """,
#)


def do_run(user, assistant, thread):
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Ben Schreck. The user has a premium account.",
    )

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "requires_action":
            action = run.required_action
            if action.type == "submit_tool_outputs":
                tool_calls = action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for call in tool_calls:
                    print(call.function.name)
                    print("ARGUMENTS")
                    print(json.loads(call.function.arguments))
                    response = requests.post(
                        FUNCTIONS_URL + f"/?function={call.function.name}",
                        json=json.loads(call.function.arguments),
                    )
                    try:
                        response.raise_for_status()
                    except requests.exceptions.HTTPError as e:
                        breakpoint()
                        print(f"Error: {e}, passing back to assistant")
                        tool_outputs.append(
                            {"tool_call_id": call.id, "output": f"Error: {e}"}
                        )
                        continue
                    try:
                        output = json.dumps(response.json())
                    except JSONDecodeError:
                        breakpoint()
                        print("bad json response", response.text)
                        output = response.text
                    tool_outputs.append({"tool_call_id": call.id, "output": output})

                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )
            else:
                raise Exception(f"unknown action type {action.type}")
        elif run.status == "completed":
            break


def run_for_user(user_id, assistant):
    thread = create_or_retrieve_user_thread(user_id)
    previously_printed_messages = set()
    while True:
        messages = client.beta.threads.messages.list(thread_id=thread.id, limit=10)
        # TODO figure out how to get next page of messages
        try:
            for message in messages.data:
                if message.id in previously_printed_messages:
                    continue
                previously_printed_messages.add(message.id)
                print(message.content.text.value)
                if message.content.text.annotations:
                    print("Annotations: ", message.content.text.annotations)
                print("")
            new_message = input("Your message: ")
            client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=new_message
            )
        except Exception as e:
            breakpoint()
            continue
        try:
            do_run(user_id, assistant, thread)
        except:
            breakpoint()
            continue

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = None
    assistant = create_or_retrieve_assistant()
    run_for_user(user_id, assistant)
