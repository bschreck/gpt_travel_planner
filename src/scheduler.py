from collections import defaultdict
from ortools.sat.python import cp_model
from shift_scheduling_app import add_soft_sequence_constraint
from itertools import permutations

class Scheduler:
    def __init__(
        self,
        ndays,
        flight_costs,
        # TODO make into a dataclass
        contiguous_sequence_constraints,
        start_city,
        end_city=None
    ):
        self.ndays = ndays
        self.start_day, self.end_day = 1, ndays
        self.flight_costs = flight_costs
        self.contiguous_sequence_constraints = contiguous_sequence_constraints
        self.start_city = start_city
        self.end_city = end_city or start_city
        self.max_visits = {city: max_visit for city, _, _, _, _, _, _, max_visit in self.contiguous_sequence_constraints}
        self.cities = set(
            [origin for origin, _ in flight_costs]
            + [destination for _, destination in flight_costs]
        )

    def get_days_in_city_var(self, city, visit, date):
        return self.days_in_city[f"{city}_{visit}_{date}"]

    def get_flight_var(self, origin, dest, day, ovisit, dvisit):
        return self.flight_vars[f"{origin}_{ovisit}_{dest}_{dvisit}_{day}"]

    def create_flight_vars(self):
        self.flight_vars = {}
        for origin, destination in self.flight_costs:
            for day in range(self.start_day, self.end_day+1):
                for ovisit in range(1, self.max_visits.get(origin,1)+1):
                    for dvisit in range(1, self.max_visits.get(destination,1)+1):
                        flight_id = f"{origin}_{ovisit}_{destination}_{dvisit}_{day}"
                        self.flight_vars[flight_id] = self.model.NewBoolVar(flight_id)

    def create_days_in_city_vars(self):
        self.days_in_city = {}
        for city in self.cities:
            for d in range(self.start_day, self.end_day+1):
                for visit in range(1, self.max_visits.get(city,1)+1):
                    var_id = f"{city}_{visit}_{d}"
                    self.days_in_city[var_id] = self.model.NewBoolVar(var_id)

    def add_contiguous_sequence_constraints(self):
        for sc in self.contiguous_sequence_constraints:
            city, hard_min, soft_min, min_cost, hard_max, soft_max, max_cost, max_visit = sc
            sequence = [self.get_days_in_city_var(city,1,d) for d in range(self.start_day, self.ndays+1)]
            # forbid sequences where the length of the longest contiguous sequence of city X is < hard_min or > hard_max
            # forbid sequences where the number of discrete contiguous sequence of city X is < hard_min or > hard_max
            # TODO: this should be at least one sequence following these constraints, but additional ones can be less than hard_min
            # and shouldn't be penalized (as much)
            variables, coeffs = add_soft_sequence_constraint(
                self.model,
                sequence,
                hard_min,
                soft_min,
                min_cost,
                soft_max,
                hard_max,
                max_cost,
                "contiguous_sequence_constraint(city %s #%d)" % (city, 1),
            )
            self.obj_bool_vars.extend(variables)
            self.obj_bool_coeffs.extend(coeffs)
            # have to be in city at least one day
            self.model.AddAtLeastOne(sequence)
            if max_visit > 1:
                for i in range(2, max_visit+1):
                    hard_min = 1
                    soft_min = 1
                    sequence = [self.get_days_in_city_var(city,i,d) for d in range(self.start_day, self.ndays+1)]
                    variables, coeffs = add_soft_sequence_constraint(
                        self.model,
                        sequence,
                        hard_min,
                        soft_min,
                        min_cost,
                        soft_max,
                        hard_max,
                        max_cost,
                        "contiguous_sequence_constraint(city %s #%d)" % (city, i),
                    )
                    self.obj_bool_vars.extend(variables)
                    self.obj_bool_coeffs.extend(coeffs)
    def add_flight_transition_constraints(self):
        for origin, destination in permutations(self.cities, 2):
            origin_max_visits = self.max_visits.get(origin, 1)
            dest_max_visits = self.max_visits.get(destination, 1)
            for day in range(self.start_day, self.ndays):
                for ovisit in range(1, origin_max_visits+1):
                    for dvisit in range(1, dest_max_visits+1):
                        origin_city_var = self.get_days_in_city_var( origin, ovisit, day)
                        dest_city_var = self.get_days_in_city_var(destination, dvisit, day+1)
                        transition = [
                            origin_city_var.Not(),
                            dest_city_var.Not(),
                        ]

                        if (origin, destination) not in self.flight_costs:
                            # Disallow transitions that don't have a flight
                            self.model.AddBoolOr(transition)
                        else:
                            flight_var = self.get_flight_var(origin, destination, day, ovisit, dvisit)
                            cost = self.flight_costs[(origin, destination)]

                            self.model.AddImplication(flight_var, origin_city_var)
                            self.model.AddImplication(flight_var, dest_city_var)

                            not_origin_or_not_dest = self.model.NewBoolVar(f'not_origin_{origin}_{day}_OR_not_dest_{destination}_{day+1})')

                            self.model.AddBoolOr([origin_city_var.Not(), dest_city_var.Not()]).OnlyEnforceIf(not_origin_or_not_dest)
                            self.model.AddImplication(flight_var.Not(), not_origin_or_not_dest)


                            self.obj_bool_vars.append(flight_var)
                            self.obj_bool_coeffs.append(cost)

    def add_singular_city_constraints(self):
        #Disallow 2 cities at once (for each day, sum of cities is 1)
        for day in range(self.start_day, self.ndays+1):
            self.model.AddExactlyOne([
                self.get_days_in_city_var(city, visit, day)
                for city in self.cities
                for visit in range(1, self.max_visits.get(city, 1)+1)
            ])
    def add_singular_flight_constraints(self):
        #Disallow more than 1 flight per day (for each day, sum of flights is 1)
        for day in range(self.start_day, self.ndays+1):
            flights_on_day = [
                self.get_flight_var(o, d, day, ovisit, dvisit)
                for o, d in self.flight_costs
                for ovisit in range(1, self.max_visits.get(o, 1)+1)
                for dvisit in range(1, self.max_visits.get(d, 1)+1)
            ]
            self.model.Add(sum(flights_on_day) <= 1)

    def limit_contiguous_subsequences(self, days_in_city, max_visits, city, visit, must_visit):
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
        for i in range(1, num_days):
            # If not in city on day i-1 and in city on day i, it's a new visit
            transition = self.model.NewBoolVar(f'transition({city},{visit},{i})')
            self.model.AddBoolAnd([days_in_city[i-1].Not(), days_in_city[i]]).OnlyEnforceIf(transition)
            transitions.append(transition)

        # Count the number of visits and add constraint
        self.model.Add(sum(transitions) <= max_visits)
        if must_visit:
            self.model.Add(sum(transitions) >= 1)
        elif max_visits > 0:
            any_days_in_city = self.model.NewBoolVar(f'{city}_{visit}_any_days')
            self.model.AddMaxEquality(any_days_in_city, days_in_city)
            self.model.Add(sum(transitions) >= 1).OnlyEnforceIf(any_days_in_city)

        return transitions



    def add_city_visits_constraints(self):
        #Disallow more than X contiguous sequences in a city
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
                visits = range(1, self.max_visits.get(city, 1)+1)
            for visit in visits:
                # TODO this can be parameterized if user wants to visit a city at least N times
                must_visit = visit == 1 and city != self.start_city
                days_in_city_vars = [
                    self.get_days_in_city_var(city,visit,day)
                    for day in range(self.start_day,self.ndays+1)
                ]
                self.transitions[(city, visit)] = self.limit_contiguous_subsequences(
                    days_in_city_vars,
                    max_visit,
                    city,
                    visit,
                    must_visit
                )
    def add_start_end_city_constraints(self):
        self.model.Add(self.get_days_in_city_var(self.start_city,1,1) == 1)
        self.model.Add(self.get_days_in_city_var(self.end_city,1,self.ndays) == 1)

    def set_objective(self):
        self.model.Minimize(
            sum(self.obj_bool_vars[i] * self.obj_bool_coeffs[i] for i in range(len(self.obj_bool_vars)))
        )

    def create_model(self):
        self.model = cp_model.CpModel()
        self.obj_bool_vars = []
        self.obj_bool_coeffs = []
        self.create_flight_vars()
        self.create_days_in_city_vars()
        self.add_contiguous_sequence_constraints()
        self.add_flight_transition_constraints()
        self.add_singular_city_constraints()
        self.add_singular_flight_constraints()
        self.add_city_visits_constraints()
        self.add_start_end_city_constraints()
        self.set_objective()

    def solve(self, verbose=True):
        solver = cp_model.CpSolver()
        if verbose:
            solver.parameters.log_search_progress = True
            solver.parameters.cp_model_presolve = False  # Disables presolve to see the search log.
            solver.parameters.cp_model_probing_level = 0  # Disables probing to see the search log.
            solver.parameters.enumerate_all_solutions = True
        solution_printer = cp_model.ObjectiveSolutionPrinter()

        status = solver.Solve(self.model, solution_printer)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            flights = []
            for varname, var in self.flight_vars.items():
                value = solver.Value(var)
                if value:
                    origin, ovisit, destination, dvisit, date = varname.split('_')
                    flights.append((origin,ovisit, destination,dvisit,int(date)))
            flights = sorted(flights, key=lambda x: x[-1])
            if len(flights) == 0:
                print("No flights found")
            for origin, ovisit, destination, dvisit, date in flights:
                print(f"Flight: {origin}#{ovisit} -> {destination}#{dvisit} on {date}")
            day_to_city = defaultdict(list)
            for varname, var in self.days_in_city.items():
                value = solver.Value(var)
                if value:
                    city, visit, day = varname.split('_')
                    day_to_city[int(day)].append((city, visit))
            for day in sorted(day_to_city.keys()):
                to_print = " || ".join(f"{day}: {city} #{visit}" for city, visit in day_to_city[day])
                print(to_print)
            for (city, visit), day_transitions in self.transitions.items():
                for i, transition in enumerate(day_transitions):
                    day = i + 1
                    print(f"Transition({city}#{visit} {day}->{day+1}): {solver.Value(transition)}")

            print("Penalties:")
            for i, var in enumerate(self.obj_bool_vars):
                if solver.BooleanValue(var):
                    penalty = self.obj_bool_coeffs[i]
                    if penalty > 0:
                        print("  %s violated, penalty=%i" % (var.Name(), penalty))
                    else:
                        print("  %s fulfilled, gain=%i" % (var.Name(), -penalty))
        elif status == cp_model.INFEASIBLE:
            print('Infeasible subsets:')
            print(solver.ResponseProto())
        else:
            print("No solution found.")

        print()
        print("Statistics")
        print("  - status          : %s" % solver.StatusName(status))
        print("  - conflicts       : %i" % solver.NumConflicts())
        print("  - branches        : %i" % solver.NumBranches())
        print("  - wall time       : %f s" % solver.WallTime())


if __name__ == '__main__':
    flight_costs = {
        ('LAX', 'MEX'): 3,
        ('MEX', 'LAX'): 3,
        ('MEX', 'OAX'): 1,
        ('OAX', 'MEX'): 1,
        ('LAX', 'MGA'): 5,
        ('MGA', 'LAX'): 5,
        ('MEX', 'MGA'): 3,
        ('MGA', 'MEX'): 3,
        ('SJD', 'LAX'): 2,
        ('LAX', 'SJD'): 2,
        ('SJD', 'MEX'): 2,
        ('MEX', 'SJD'): 2,
    }


    # (city, hard_min, soft_min, min_cost, hard_max, soft_max, max_cost, max_visits)
    contiguous_sequence_constraints = [
        ('MEX', 7, 10, 100, 10, 7, 100, 2),
        ('OAX', 5, 7, 100, 7,  7, 100, 1),
        ('MGA', 6, 7, 100, 7,  7, 100, 1),
        ('SJD', 3, 4, 100, 5,  4, 100, 1),
    ]

    start_city, end_city = 'LAX', 'LAX'
    ndays = 31
    scheduler = Scheduler(
        ndays,
        flight_costs,
        contiguous_sequence_constraints,
        start_city,
        end_city
    )
    scheduler.create_model()
    scheduler.solve()
