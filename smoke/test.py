from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpInteger, LpStatus, value

# Create the model
model = LpProblem("Cake Bakery Optimization", LpMaximize)

# Define decision variables
chocolate_cakes = LpVariable('Chocolate_Cakes', lowBound=0, cat=LpInteger)
vanilla_cakes = LpVariable('Vanilla_Cakes', lowBound=0, cat=LpInteger)

# Define the objective function
profit = 12 * chocolate_cakes + 9 * vanilla_cakes
model += profit, "Total Profit"

# Define constraints
# Constraint 1: Flour constraint
flour_constraint = 2 * chocolate_cakes + 1.5 * vanilla_cakes <= 300
model += flour_constraint, "Flour Constraint"

# Constraint 2: Egg constraint
egg_constraint = 3 * chocolate_cakes + 2 * vanilla_cakes <= 400
model += egg_constraint, "Egg Constraint"

# Constraint 3: Oven capacity constraint
oven_constraint = chocolate_cakes + vanilla_cakes <= 120
model += oven_constraint, "Oven Capacity Constraint"

# Constraint 4: Market demand constraint
market_demand_constraint = vanilla_cakes >= 30
model += market_demand_constraint, "Market Demand Constraint"

# Solve the model
model.solve()

# Print the solve status
print("Status:", LpStatus[model.status])

# Print the optimal objective value
print("Optimal Profit:", value(profit))

# Print the value of each variable
print("Chocolate Cakes:", value(chocolate_cakes))
print("Vanilla Cakes:", value(vanilla_cakes))