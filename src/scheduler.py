from collections import defaultdict
from ortools.sat.python import cp_model
from itertools import permutations
from get_flight_data import build_flight_costs, build_flight_costs_from_remote_file, iata_codes_from_file
from shift_scheduling_app import negated_bounded_span
from dataclasses import dataclass
from config import DEFAULT_BUCKET, DEFAULT_FLIGHTS_FILE
from tqdm import tqdm
import pandas as pd
import pickle
import networkx as nx
from networkx import all_pairs_dijkstra_path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def add_soft_sequence_constraint(
    model, works, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost, prefix
):
    """Sequence constraint on true variables with soft and hard bounds.

    This constraint look at every maximal contiguous sequence of variables
    assigned to true. If forbids sequence of length < hard_min or > hard_max.
    Then it creates penalty terms if the length is < soft_min or > soft_max.

    Args:
      model: the sequence constraint is built on this model.
      works: a list of Boolean variables.
      hard_min: any sequence of true variables must have a length of at least
        hard_min.
      soft_min: any sequence should have a length of at least soft_min, or a
        linear penalty on the delta will be added to the objective.
      min_cost: the coefficient of the linear penalty if the length is less than
        soft_min.
      soft_max: any sequence should have a length of at most soft_max, or a linear
        penalty on the delta will be added to the objective.
      hard_max: any sequence of true variables must have a length of at most
        hard_max.
      max_cost: the coefficient of the linear penalty if the length is more than
        soft_max.
      prefix: a base name for penalty literals.

    Returns:
      a tuple (variables_list, coefficient_list) containing the different
      penalties created by the sequence constraint.
    """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    if hard_min is not None:
        for length in range(1, hard_min):
            for start in range(len(works) - length + 1):
                model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0 and soft_min is not None:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ": under_span(start=%i, length=%i)" % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0 and soft_max is not None:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ": over_span(start=%i, length=%i)" % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    if hard_max is not None:
        for start in range(len(works) - hard_max):
            model.AddBoolOr(
                [works[i].Not() for i in range(start, start + hard_max + 1)]
            )
    return cost_literals, cost_coefficients


@dataclass
class ContiguousSequenceConstraint:
    city: str
    hard_min: float = None
    hard_max: float = None
    soft_min: float = None
    soft_max: float = None
    min_cost: float = 100
    max_cost: float = 100
    max_visits: int = 1

    def __post_init__(self):
        if (
            self.hard_min is None
            and self.hard_max is None
            and self.soft_min is None
            and self.soft_max is None
        ):
            raise ValueError(
                "Must provide at least one of hard_min, hard_max, soft_min, soft_max"
            )
        if self.soft_min is None:
            self.soft_min = self.hard_min
        if self.hard_min is None:
            self.hard_min = self.soft_min
        if self.soft_max is None:
            self.soft_max = self.hard_max
        if self.hard_max is None:
            self.hard_max = self.soft_max


@dataclass
class DateRangeConstraint:
    city: str
    min_start_day: int | None = None
    max_start_day: int | None = None
    min_end_day: int | None = None
    max_end_day: int | None = None
    visit: int = 1

    def __post_init__(self):
        if (
            self.min_start_day is None
            and self.max_start_day is None
            and self.min_end_day is None
            and self.max_end_day is None
        ):
            raise ValueError(
                "Must provide at least one of min_start_day, max_start_day, min_end_day, or max_end_day"
            )


class Scheduler:
    def __init__(
        self,
        ndays: int,
        flight_costs: dict[tuple[str, str], float],
        start_city: str,
        contiguous_sequence_constraints: list[ContiguousSequenceConstraint]
        | None = None,
        date_range_constraints: list[DateRangeConstraint] | None = None,
        relevant_cities: list[str] | None = None,
        end_city: str | None = None,
        must_visits: list[str] | None = None,
    ):
        # TODO get more data in flight costs
        airport_mapping = {
            'AEP': 'EZE'
        }
        for k, v in airport_mapping.items():
            for (o, d), c in flight_costs.items():
                if o == k:
                    flight_costs[(v, d)] = c
                if d == k:
                    flight_costs[(o, v)] = c
        self.ndays = ndays
        self.start_day, self.end_day = 1, ndays
        if (
            contiguous_sequence_constraints is None
            and date_range_constraints is None
            and relevant_cities is None
        ):
            raise ValueError(
                "Must provide at least one of contiguous_sequence_constraints, date_range_constraints, or relevant_cities"
            )
        self.contiguous_sequence_constraints = contiguous_sequence_constraints or []
        self.date_range_constraints = date_range_constraints or []
        self.start_city = start_city
        self.end_city = end_city or start_city

        self.cities = set(relevant_cities or [])
        self.cities = self.cities | set([self.start_city, self.end_city])
        self.cities = self.cities | set(
            sc.city for sc in self.contiguous_sequence_constraints
        )
        self.cities = self.cities | set(drc.city for drc in self.date_range_constraints)

        # These default to 1
        self.max_visits = {
            sc.city: sc.max_visits for sc in self.contiguous_sequence_constraints
        }
        # self.flight_costs = get_approx_flight_data(flight_costs, self.cities)

        # remove irrelevant cities

        self.flight_costs = {
            k: v
            for k, v in flight_costs.items()
            if k[0] in self.cities and k[1] in self.cities
        }

        for o, d in self.flight_costs:
            self.cities.add(o)
            self.cities.add(d)

        self.must_visits = set(must_visits or [])
        for sc in self.contiguous_sequence_constraints:
            self.must_visits.add(sc.city)
        for drc in self.date_range_constraints:
            self.must_visits.add(drc.city)

    def get_days_in_city_var(self, city, visit, date):
        return self.days_in_city[f"{city}_{visit}_{date}"]

    def get_flight_var(self, origin, dest, day, ovisit, dvisit):
        return self.flight_vars[f"{origin}_{ovisit}_{dest}_{dvisit}_{day}"]

    def create_flight_vars(self):
        self.flight_vars = {}
        for origin, destination in self.flight_costs:
            for day in range(self.start_day, self.end_day + 1):
                for ovisit in range(1, self.max_visits.get(origin, 1) + 1):
                    for dvisit in range(1, self.max_visits.get(destination, 1) + 1):
                        flight_id = f"{origin}_{ovisit}_{destination}_{dvisit}_{day}"
                        self.flight_vars[flight_id] = self.model.NewBoolVar(flight_id)

    def create_days_in_city_vars(self):
        self.days_in_city = {}
        for city in self.cities:
            for d in range(self.start_day, self.end_day + 1):
                for visit in range(1, self.max_visits.get(city, 1) + 1):
                    var_id = f"{city}_{visit}_{d}"
                    self.days_in_city[var_id] = self.model.NewBoolVar(var_id)

    def add_date_range_constraints(self):
        for drc in self.date_range_constraints:
            for day in range(self.start_day + 1, self.ndays):
                var = self.get_days_in_city_var(drc.city, drc.visit, day)
                if drc.min_start_day is not None and drc.min_start_day > day:
                    self.model.Add(var == 0)

                if drc.max_start_day is not None and drc.max_start_day == day:
                    # We should be in city at least one day before or equal to max_start_day
                    prior_days = [
                        self.get_days_in_city_var(drc.city, drc.visit, d)
                        for d in range(self.start_day, day + 1)
                    ]
                    self.model.AddBoolOr(prior_days)

                ## if we have both a max_start and min_end, we need to be in city
                ## all days in between
                if (
                    drc.max_start_day is not None
                    and drc.max_start_day <= day
                    and drc.min_end_day is not None
                    and drc.min_end_day >= day
                ):
                    self.model.Add(var == 1)

                ## otherwise with a min_end_day we have to at least be in city on min_end_day
                elif drc.min_end_day is not None and drc.min_end_day == day:
                    self.model.Add(var == 1)

                ## We should not be in city after max_end_day
                if drc.max_end_day is not None and drc.max_end_day < day:
                    self.model.Add(var == 0)

    def add_contiguous_sequence_constraints(self):
        for sc in self.contiguous_sequence_constraints:
            sequence = [
                self.get_days_in_city_var(sc.city, 1, d)
                for d in range(self.start_day, self.ndays + 1)
            ]
            # forbid sequences where the length of the longest contiguous sequence of city X is < hard_min or > hard_max
            # forbid sequences where the number of discrete contiguous sequence of city X is < hard_min or > hard_max
            # TODO: this should be at least one sequence following these constraints, but additional ones can be less than hard_min
            # and shouldn't be penalized (as much)
            variables, coeffs = add_soft_sequence_constraint(
                self.model,
                sequence,
                sc.hard_min,
                sc.soft_min,
                sc.min_cost,
                sc.soft_max,
                sc.hard_max,
                sc.max_cost,
                "contiguous_sequence_constraint(city %s #%d)" % (sc.city, 1),
            )
            self.obj_bool_vars.extend(variables)
            self.obj_bool_coeffs.extend(coeffs)
            # have to be in city at least one day
            self.model.AddAtLeastOne(sequence)
            if sc.max_visits > 1:
                for i in range(2, sc.max_visits + 1):
                    hard_min = 1
                    soft_min = 1
                    sequence = [
                        self.get_days_in_city_var(sc.city, i, d)
                        for d in range(self.start_day, self.ndays + 1)
                    ]
                    variables, coeffs = add_soft_sequence_constraint(
                        self.model,
                        sequence,
                        hard_min,
                        soft_min,
                        sc.min_cost,
                        sc.soft_max,
                        sc.hard_max,
                        sc.max_cost,
                        "contiguous_sequence_constraint(city %s #%d)" % (sc.city, i),
                    )
                    self.obj_bool_vars.extend(variables)
                    self.obj_bool_coeffs.extend(coeffs)

    def add_flight_transition_constraints(self):
        for origin, destination in permutations(self.cities, 2):
            origin_max_visits = self.max_visits.get(origin, 1)
            dest_max_visits = self.max_visits.get(destination, 1)
            for day in range(self.start_day, self.ndays):
                for ovisit in range(1, origin_max_visits + 1):
                    for dvisit in range(1, dest_max_visits + 1):
                        origin_city_var = self.get_days_in_city_var(origin, ovisit, day)
                        dest_city_var = self.get_days_in_city_var(
                            destination, dvisit, day + 1
                        )
                        transition = [origin_city_var.Not(), dest_city_var.Not()]

                        if (origin, destination) not in self.flight_costs:
                            # Disallow transitions that don't have a flight
                            self.model.AddBoolOr(transition)
                        else:
                            flight_var = self.get_flight_var(
                                origin, destination, day, ovisit, dvisit
                            )
                            cost = self.flight_costs[(origin, destination)]

                            self.model.AddImplication(flight_var, origin_city_var)
                            self.model.AddImplication(flight_var, dest_city_var)

                            not_origin_or_not_dest = self.model.NewBoolVar(
                                f"not_origin_{origin}_{day}_OR_not_dest_{destination}_{day+1})"
                            )

                            self.model.AddBoolOr(
                                [origin_city_var.Not(), dest_city_var.Not()]
                            ).OnlyEnforceIf(not_origin_or_not_dest)
                            self.model.AddImplication(
                                flight_var.Not(), not_origin_or_not_dest
                            )

                            self.obj_bool_vars.append(flight_var)
                            self.obj_bool_coeffs.append(cost)

    def add_singular_city_constraints(self):
        # Disallow 2 cities at once (for each day, sum of cities is 1)
        for day in range(self.start_day, self.ndays + 1):
            self.model.AddExactlyOne(
                [
                    self.get_days_in_city_var(city, visit, day)
                    for city in self.cities
                    for visit in range(1, self.max_visits.get(city, 1) + 1)
                ]
            )

    def add_singular_flight_constraints(self):
        # Disallow more than 1 flight per day (for each day, sum of flights is 1)
        for day in range(self.start_day, self.ndays + 1):
            flights_on_day = [
                self.get_flight_var(o, d, day, ovisit, dvisit)
                for o, d in self.flight_costs
                for ovisit in range(1, self.max_visits.get(o, 1) + 1)
                for dvisit in range(1, self.max_visits.get(d, 1) + 1)
            ]
            self.model.Add(sum(flights_on_day) <= 1)

    def limit_contiguous_subsequences(
        self, days_in_city, max_visits, city, visit, must_visit
    ):
        """
        Adds constraints to the model to limit the number of discrete contiguous subsequences.

        Args:
            model: The CP-SAT model.
            days_in_city: A list of BoolVar where each BoolVar represents presence in a city on a day.
            max_visits: The maximum allowed number of visits (discrete contiguous subsequences).
        """

        transitions = []
        num_days = len(days_in_city)

        # Identify transitions from 'not in city' to 'in city'
        for day in range(2, num_days + 1):
            flight_vars_to_city_on_day = [
                self.get_flight_var(origin, city, day - 1, ovisit, visit)
                for origin in self.cities
                for ovisit in range(1, self.max_visits.get(origin, 1) + 1)
                if origin != city
            ]

            # If any flight to city on day - 1, then transition to city on day
            transition = self.model.NewBoolVar(f"transition({city},{visit},{day})")
            self.model.AddMaxEquality(transition, flight_vars_to_city_on_day)
            transitions.append(transition)

        # Count the number of visits and add constraint
        if must_visit:
            self.model.Add(sum(transitions) == 1)
        elif max_visits > 0:
            any_days_in_city = self.model.NewBoolVar(f"{city}_{visit}_any_days")
            self.model.AddMaxEquality(any_days_in_city, days_in_city)
            self.model.Add(sum(transitions) == 1).OnlyEnforceIf(any_days_in_city)
            # TODO: end_city should be counted as visit 2 for start city if they are equal?

        return transitions

    def add_city_visits_constraints(self):
        # Disallow more than X contiguous sequences in a city
        self.transitions = {}
        for city in self.cities:
            if city == self.start_city:
                max_visit = 0
                visits = [1]
            elif city == self.end_city:
                max_visit = 0
                visits = [1]
            else:
                max_visit = 1
                visits = range(1, self.max_visits.get(city, 1) + 1)
            for visit in visits:
                # TODO this can be parameterized if user wants to visit a city at least N times
                # TODO do we need a check on not end_city too?
                must_visit = city in self.must_visits
                days_in_city_vars = [
                    self.get_days_in_city_var(city, visit, day)
                    for day in range(self.start_day, self.ndays + 1)
                ]
                self.transitions[(city, visit)] = self.limit_contiguous_subsequences(
                    days_in_city_vars, max_visit, city, visit, must_visit
                )

    def add_start_end_city_constraints(self):
        self.model.Add(self.get_days_in_city_var(self.start_city, 1, 1) == 1)
        self.model.Add(self.get_days_in_city_var(self.end_city, 1, self.ndays) == 1)

    def set_objective(self):
        self.model.Minimize(
            sum(
                self.obj_bool_vars[i] * self.obj_bool_coeffs[i]
                for i in range(len(self.obj_bool_vars))
            )
        )

    def create_model(self):
        self.model = cp_model.CpModel()
        self.obj_bool_vars = []
        self.obj_bool_coeffs = []
        self.create_flight_vars()
        self.create_days_in_city_vars()
        self.add_contiguous_sequence_constraints()
        self.add_date_range_constraints()
        self.add_flight_transition_constraints()
        self.add_singular_city_constraints()
        self.add_singular_flight_constraints()
        self.add_city_visits_constraints()
        self.add_start_end_city_constraints()
        self.set_objective()

    def solve(self, verbose=True):
        self.solver = cp_model.CpSolver()
        if verbose:
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.cp_model_presolve = (
                False  # Disables presolve to see the search log.
            )
            self.solver.parameters.cp_model_probing_level = (
                0  # Disables probing to see the search log.
            )
            self.solver.parameters.enumerate_all_solutions = True
        solution_printer = cp_model.ObjectiveSolutionPrinter()

        self.status = self.solver.Solve(self.model, solution_printer)
        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            flights = []
            for varname, var in self.flight_vars.items():
                value = self.solver.Value(var)
                if value:
                    origin, ovisit, destination, dvisit, date = varname.split("_")
                    flights.append((origin, ovisit, destination, dvisit, int(date)))
            flights = sorted(flights, key=lambda x: x[-1])
            if len(flights) == 0:
                print("No flights found")
            for origin, ovisit, destination, dvisit, date in flights:
                print(f"Flight: {origin}#{ovisit} -> {destination}#{dvisit} on {date}")
            day_to_city = defaultdict(list)
            for varname, var in self.days_in_city.items():
                value = self.solver.Value(var)
                if value:
                    city, visit, day = varname.split("_")
                    day_to_city[int(day)].append((city, visit))
            for day in sorted(day_to_city.keys()):
                to_print = " || ".join(
                    f"{day}: {city} #{visit}" for city, visit in day_to_city[day]
                )
                print(to_print)
            for (city, visit), day_transitions in self.transitions.items():
                for i, transition in enumerate(day_transitions):
                    day = i + 1
                    print(
                        f"Transition({city}#{visit} {day}->{day+1}): {self.solver.Value(transition)}"
                    )

            print("Penalties:")
            for i, var in enumerate(self.obj_bool_vars):
                if self.solver.BooleanValue(var):
                    penalty = self.obj_bool_coeffs[i]
                    if penalty > 0:
                        print("  %s violated, penalty=%i" % (var.Name(), penalty))
                    else:
                        print("  %s fulfilled, gain=%i" % (var.Name(), -penalty))
        elif self.status == cp_model.INFEASIBLE:
            print("Infeasible subsets:")
            print(self.solver.ResponseProto())
        else:
            print("No solution found.")

        print()
        print("Statistics")
        print("  - status          : %s" % self.solver.StatusName(self.status))
        print("  - conflicts       : %i" % self.solver.NumConflicts())
        print("  - branches        : %i" % self.solver.NumBranches())
        print("  - wall time       : %f s" % self.solver.WallTime())

    def get_result_flight_records(self):
        if self.status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return []
        flights = []
        for varname, var in self.flight_vars.items():
            value = self.solver.Value(var)
            if value:
                origin, _, destination, _, day = varname.split("_")
                flights.append(
                    {"origin": origin, "destination": destination, "day": int(day)}
                )
        return sorted(flights, key=lambda x: x["day"])

def get_approx_flight_data(nonstop_flight_costs, layover_time=2):
    G = nx.Graph()
    nonstop_flight_costs = {
        (o, d): c
        for (o, d), c in nonstop_flight_costs.items()
        if o is not None and d is not None
    }
    G.add_weighted_edges_from([(o, d, c) for (o, d), c in nonstop_flight_costs.items()])
    paths = all_pairs_dijkstra_path(G, weight='weight')
    def path_to_cost(path):
        layover_cost = layover_time * (len(path) - 2)
        return layover_cost + sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    expanded_dict_flights = {
        (o, d): path_to_cost(path) for o, d_to_path in paths
        for d, path in d_to_path.items()
        if o != d
    }
    return expanded_dict_flights


def get_approx_flight_data_scipy(nonstop_flight_costs, layover_time=2):
    nodes = set()
    for (o, d), _ in nonstop_flight_costs.items():
        nodes.update([o, d])
    nodes = sorted(list(nodes))
    node_indices = {node: i for i, node in enumerate(nodes)}

    num_nodes = len(nodes)
    graph_matrix = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(graph_matrix, 0)

    for (o, d), cost in nonstop_flight_costs.items():
        o_idx, d_idx = node_indices[o], node_indices[d]
        graph_matrix[o_idx][d_idx] = cost

    sparse_matrix = csr_matrix(graph_matrix)

    distances, predecessors = dijkstra(csgraph=sparse_matrix, directed=False, return_predecessors=True, indices=None)

    def get_path(Pr, i, j):
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        return path[::-1]

    expanded_dict_flights = {}
    for o in nodes:
        o_idx = node_indices[o]
        for d in nodes:
            d_idx = node_indices[d]
            if o != d and distances[o_idx, d_idx] != np.inf:
                path_indices = get_path(predecessors, o_idx, d_idx)
                path = [nodes[idx] for idx in path_indices]
                layover_cost = layover_time * (len(path) - 2)
                total_cost = distances[o_idx, d_idx] + layover_cost
                expanded_dict_flights[(o, d)] = total_cost
    return expanded_dict_flights



@dataclass
class ScheduleTripParams:
    start_city: str
    ndays: int
    bucket: str = DEFAULT_BUCKET
    filename: str = DEFAULT_FLIGHTS_FILE
    contiguous_sequence_constraints: list[ContiguousSequenceConstraint] | None = None
    date_range_constraints: list[DateRangeConstraint] | None = None
    end_city: str | None = None
    relevant_cities: list[str] | None = None
    must_visits: list[str] | None = None


def parse_schedule_trip_json(request_json: dict) -> ScheduleTripParams:
    iata_codes = set(iata_codes_from_file()['iata_code'].tolist())
    bad_cities = set()
    passed_cities = set(
        [request_json['start_city']]
        + [request_json.get('end_city', None)]
        + request_json.get('relevant_cities', [])
        + request_json.get('must_visits', [])
        + [sc_raw['city'] for sc_raw in request_json.get("contiguous_sequence_constraints", [])]
        + [drc_raw['city'] for drc_raw in request_json.get("date_range_constraints", [])]
    ) - set([None])
    for city in passed_cities:
        if city not in iata_codes:
            bad_cities.add(city)
    if len(bad_cities) > 0:
        raise ValueError(f"Invalid iata airport codes: {bad_cities}")

    start_city = request_json["start_city"]
    end_city = request_json.get("end_city", None)
    ndays = request_json["ndays"]
    bucket = request_json.get("bucket", DEFAULT_BUCKET)
    filename = request_json.get("filename", DEFAULT_FLIGHTS_FILE)
    contiguous_sequence_constraints = []
    relevant_cities = request_json.get("relevant_cities", [])
    must_visits = request_json.get("must_visits", [])
    for sc_raw in request_json.get("contiguous_sequence_constraints", []):
        contiguous_sequence_constraints.append(
            ContiguousSequenceConstraint(
                city=sc_raw["city"],
                hard_min=sc_raw.get("hard_min", None),
                soft_min=sc_raw.get("soft_min", None),
                soft_max=sc_raw.get("soft_max", None),
                hard_max=sc_raw.get("hard_max", None),
                min_cost=sc_raw.get("min_cost", 100),
                max_cost=sc_raw.get("max_cost", 100),
                max_visits=sc_raw.get("max_visits", 1),
            )
        )
    date_range_constraints = []
    for drc_raw in request_json.get("date_range_constraints", []):
        date_range_constraints.append(
            DateRangeConstraint(
                city=drc_raw["city"],
                min_start_day=drc_raw.get("min_start_day", None),
                max_start_day=drc_raw.get("max_start_day", None),
                min_end_day=drc_raw.get("min_end_day", None),
                max_end_day=drc_raw.get("max_end_day", None),
                visit=drc_raw.get("visit", 1),
            )
        )

    return ScheduleTripParams(
        start_city=start_city,
        end_city=end_city,
        ndays=ndays,
        bucket=bucket,
        filename=filename,
        contiguous_sequence_constraints=contiguous_sequence_constraints,
        date_range_constraints=date_range_constraints,
        relevant_cities=relevant_cities,
        must_visits=must_visits,
    )


if __name__ == "__main__":
    bucket = "gpt-travel-planner-data"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    flight_costs = build_flight_costs_from_remote_file(
        bucket, f"{today}/flights.pickle", f"{today}_flights.pickle"
    )
    #  with open("flights_full.pickle", "rb") as f:
    #  flights = pickle.load(f)
    #  flight_costs = build_flight_costs(flights)

    contiguous_sequence_constraints = [
        ContiguousSequenceConstraint(
            city="MEX",
            hard_min=7,
            soft_min=10,
            min_cost=100,
            hard_max=10,
            soft_max=7,
            max_cost=100,
            max_visits=2,
        ),
        ContiguousSequenceConstraint(
            city="OAX",
            hard_min=5,
            soft_min=7,
            min_cost=100,
            hard_max=7,
            soft_max=7,
            max_cost=100,
            max_visits=1,
        ),
        ContiguousSequenceConstraint(
            city="MGA",
            hard_min=6,
            soft_min=7,
            min_cost=100,
            hard_max=7,
            soft_max=7,
            max_cost=100,
            max_visits=1,
        ),
        ContiguousSequenceConstraint(
            city="SJD",
            hard_min=3,
            soft_min=4,
            min_cost=100,
            hard_max=5,
            soft_max=4,
            max_cost=100,
            max_visits=1,
        ),
    ]

    date_range_constraints = [
        # TODO soft/hard
        DateRangeConstraint(city="MEX", min_start_day=2, max_end_day=20, visit=1),
        DateRangeConstraint(
            city="MGA",
            min_start_day=16,
            max_start_day=17,
            min_end_day=23,
            max_end_day=24,
        ),
    ]

    start_city, end_city = "LAX", "LAX"

    ndays = 31
    scheduler = Scheduler(
        ndays=ndays,
        flight_costs=flight_costs,
        contiguous_sequence_constraints=contiguous_sequence_constraints,
        start_city=start_city,
        end_city=end_city,
        date_range_constraints=date_range_constraints,
    )
    scheduler.create_model()
    scheduler.solve()
