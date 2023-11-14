tools = [
    {
        "type": "function",
        "function": {
            "name": "pick_flights",
            "description": "Pick flights for a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {"type": "string"},
                    "max_flights": {
                        "type": "integer",
                        "description": "The maximum number of flights to return",
                    },
                    "flight_legs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "origin",
                                "destination",
                                "earliest_departure_time",
                                "latest_arrival_time",
                            ],
                            "properties": {
                                "origin": {
                                    "type": "string",
                                    "description": "airport iata code, e.g. 'LAX'",
                                },
                                "destination": {
                                    "type": "string",
                                    "description": "airport iata code, e.g. 'LAX'",
                                },
                                "earliest_departure_time": {
                                    "type": "string",
                                    "description": "ISO 8601 format w/ time zone, e.g. '2024-01-07T08:00:00-08:00'",
                                },
                                "latest_arrival_time": {
                                    "type": "string",
                                    "description": "ISO 8601 format w/ time zone, e.g. '2024-01-07T08:00:00-08:00'",
                                },
                            },
                        },
                    },
                },
                "required": ["passenger_name", "flight_legs", "max_flights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_trip",
            "description": "Schedule flights for a user's trip subject to various constraints",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_city": {
                        "type": "string",
                        "description": "Airport iata code, e.g. 'LAX'",
                    },
                    "end_city": {
                        "type": "string",
                        "description": "Airport iata code, e.g. 'LAX'. Defaults to start_city",
                    },
                    "ndays": {"type": "integer", "description": "Total number of days"},
                    "contiguous_sequence_constraints": {
                        "type": "array",
                        "required": [
                            "city",
                            "hard_min",
                            "soft_min",
                            "soft_max",
                            "hard_max",
                            "max_visits",
                        ],
                        "items": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "Airport iata code, e.g. 'LAX'",
                                },
                                "hard_min": {
                                    "type": "number",
                                    "description": "The minimum number of days to spend in the city.",
                                },
                                "soft_min": {
                                    "type": "number",
                                    "description": "The minimum number of days to spend in the city, if possible.",
                                },
                                "soft_max": {
                                    "type": "number",
                                    "description": "The maximum number of days to spend in the city, if possible.",
                                },
                                "hard_max": {
                                    "type": "number",
                                    "description": "The maximum number of days to spend in the city.",
                                },
                                "max_visits": {
                                    "type": "integer",
                                    "description": "The maximum number of visits to the city.",
                                },
                            },
                        },
                        "description": "A list of constraints on the number of days spent in a city",
                    },
                    "relevant_cities": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Airport iata code, e.g. 'LAX'",
                        },
                        "description": "A list of cities that the user wants to, but doesn't need to, visit. You should include all possible cities in this list, because the scheduler won't look for additional ones",
                    },
                    "must_visits" : {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Airport iata code, e.g. 'LAX'",
                        },
                        "description": "A list of cities that the user must visit",
                    },
                    "date_range_constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "city",
                                "min_start_day",
                                "max_start_day",
                                "min_end_day",
                                "max_end_day",
                                "visit",
                            ],
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "Airport iata code, e.g. 'LAX'",
                                },
                                "min_start_day": {
                                    "type": "integer",
                                    "description": "Between 1 and ndays",
                                },
                                "max_start_day": {
                                    "type": "integer",
                                    "description": "Between 1 and ndays",
                                },
                                "min_end_day": {
                                    "type": "integer",
                                    "description": "Between 1 and ndays",
                                },
                                "max_end_day": {
                                    "type": "integer",
                                    "description": "Between 1 and ndays",
                                },
                                "visit": {
                                    "type": "integer",
                                    "description": "Visit number if visiting a city multiple times (first visit is 1)",
                                },
                            },
                            "description": "A list of constraints on the ordering of city visits on the trip",
                        },
                    },
                },
                "required": [
                    "start_city",
                    "ndays",
                    "contiguous_sequence_constraints",
                    "date_range_constraints",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_flight_preferences",
            "description": "Set flight preferences for a passenger. Only fields provided will be updated.",
            "parameters": {
                "type": "object",
                "required": ["passenger_name", "preferences"],
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    },
                    "preferences": {
                        "type": "object",
                        "required": [],
                        "properties": {
                            "time_of_day_order": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "early_morning",
                                        "morning",
                                        "afternoon",
                                        "evening",
                                        "red_eye",
                                    ],
                                },
                                "description": "The order of preferred times of day to fly. Must include all 5 items",
                            },
                            "hard_max_cost": {
                                "type": "number",
                                "description": "The maximum cost of a flight.",
                            },
                            "soft_max_cost": {
                                "type": "number",
                                "description": "The maximum cost of a flight, if possible.",
                            },
                            "single_leg_hard_max_cost": {
                                "type": "number",
                                "description": "The maximum cost of a single leg of a flight.",
                            },
                            "single_leg_soft_max_cost": {
                                "type": "number",
                                "description": "The maximum cost of a single leg of a flight, if possible.",
                            },
                            "soft_max_duration": {
                                "type": "integer",
                                "description": "The maximum duration of a flight in seconds, if possible.",
                            },
                            "hard_max_duration": {
                                "type": "integer",
                                "description": "The maximum duration of a flight in seconds.",
                            },
                            "soft_max_stops": {
                                "type": "integer",
                                "description": "The maximum number of stops in a flight, if possible.",
                            },
                            "hard_max_stops": {
                                "type": "integer",
                                "description": "The maximum number of stops in a flight.",
                            },
                            "soft_min_layover_duration": {
                                "type": "integer",
                                "description": "The minimum layover duration in seconds, if possible.",
                            },
                            "hard_min_layover_duration": {
                                "type": "integer",
                                "description": "The minimum layover duration in seconds.",
                            },
                            "soft_max_layover_duration": {
                                "type": "integer",
                                "description": "The maximum layover duration in seconds, if possible.",
                            },
                            "hard_max_layover_duration": {
                                "type": "integer",
                                "description": "The maximum layover duration in seconds.",
                            },
                            "airline_preferences": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "The iata code of the airline, e.g. AM, UA, B6",
                                },
                            },
                            "seat_class_preferences": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "economy",
                                        "premium_economy",
                                        "business",
                                        "first",
                                    ],
                                },
                            },
                            "seat_location_preference": {
                                "type": "string",
                                "enum": ["window", "aisle"],
                            },
                            "desires_extra_legroom": {"type": "boolean"},
                            "total_cost_weight": {
                                "type": "number",
                                "description": "The weight to assign to minimize total cost, between 0 and 1",
                            },
                            "total_duration_weight": {
                                "type": "number",
                                "description": "The weight to assign to minimize total duration, between 0 and 1",
                            },
                            "preferred_airline_ratio_weight": {
                                "type": "number",
                                "description": "The weight to assign to the ratio of preferred airlines to total flights, between 0 and 1",
                            },
                            "time_of_day_weight": {
                                "type": "number",
                                "description": "The weight to assign to preferred flight time, between 0 and 1",
                            },
                            "layover_duration_weight": {
                                "type": "number",
                                "description": "The weight to assign to preferred layover duration, between 0 and 1",
                            },
                            "nstops_weight": {
                                "type": "number",
                                "description": "The weight to assign to preferred number of stops, between 0 and 1",
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_passenger_info",
            "description": "Set passenger information. Only fields provided will be updated.",
            "parameters": {
                "type": "object",
                "required": ["passenger_name", "passenger_info"],
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    },
                    "passenger_info": {
                        "type": "object",
                        "required": [],
                        "properties": {
                            "age": {
                                "type": "integer",
                                "description": "The age of the passenger",
                            },
                            "ptype": {
                                "type": "string",
                                "enum": [
                                    "adult",
                                    "child",
                                    "infant",
                                    "senior",
                                    "student",
                                    "youth",
                                ],
                                "description": "The passenger type. Generally best to leave out and use age instead",
                            },
                            "phone_number": {
                                "type": "string",
                                "description": "The phone number of the passenger",
                            },
                            "email": {
                                "type": "string",
                                "description": "The email of the passenger",
                            },
                            "date_of_birth": {
                                "type": "string",
                                "description": "The date of birth of the passenger, in ISO 8601 format",
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_passenger_info",
            "description": "Get passenger information",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_preferences",
            "description": "Get flight preferences for a passenger",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    }
                },
            },
        },
    },
]
