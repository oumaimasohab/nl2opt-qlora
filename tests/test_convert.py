import pytest

from nl2opt.convert import IncompleteFormulation, identifiers, parse, parse_number, solve, to_program
from nl2opt.execute import run_program


@pytest.mark.parametrize("text, value", [
    ("3,200", 3200), ("$500", 500), ("2$.", 2), ("5%", 0.05), ("35 percent", 0.35), ("thirty percent", 0.3),
    ("five", 5), ("one hundred thousand", 100000), ("half", 0.5), ("a third", 1 / 3), ("one third", 1 / 3),
    ("twice", 2), ("thrice", 3), ("3 times", 3), ("three times", 3), ("8000 minutes", 8000), ("1.8$.", 1.8),
])
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


def test_parse_number_rejects_garbage():
    with pytest.raises(ValueError):
        parse_number("lots")


def record(constraints, obj_terms=None, direction="maximize", document="", obj_vars=None, vars_=("x", "y")):
    obj = {"type": "objective", "direction": direction, "name": "profit"}
    if obj_vars is not None:
        obj["vars"] = obj_vars
    else:
        obj["terms"] = obj_terms or {"x": "3", "y": "5"}
    return {"vars": list(vars_), "order_mapping": {v: i for i, v in enumerate(vars_)} | {"X item": 0},
            "obj_declaration": obj, "const_declarations": constraints, "document": document}


LE, GE = "LESS_OR_EQUAL", "GREATER_OR_EQUAL"


@pytest.mark.parametrize("constraint, coefs, op, rhs", [
    ({"type": "linear", "terms": {"X item": "2", "y": "1"}, "limit": "10", "direction": "at most", "operator": LE},
     {"x": 2, "y": 1}, "<=", 10),
    ({"type": "sum", "limit": "100", "direction": "at least", "operator": GE}, {"x": 1, "y": 1}, ">=", 100),
    ({"type": "lowerbound", "var": "x", "limit": "5", "direction": "at least", "operator": GE}, {"x": 1}, ">=", 5),
    ({"type": "upperbound", "var": "y", "limit": "7", "direction": "at most", "operator": LE}, {"y": 1}, "<=", 7),
    ({"type": "xby", "x_var": "x", "y_var": "y", "param": "twice", "direction": "at least", "operator": GE},
     {"x": 1, "y": -2}, ">=", 0),
    ({"type": "xy", "x_var": "x", "y_var": "y", "direction": "at most", "operator": LE}, {"x": 1, "y": -1}, "<=", 0),
    ({"type": "ratio", "var": "x", "limit": "30%", "direction": "at least", "operator": GE},
     {"x": 0.7, "y": -0.3}, ">=", 0),
    ({"type": "ratio", "var": "x", "limit": "25", "direction": "at most", "operator": LE},  # bare number = percent
     {"x": 0.75, "y": -0.25}, "<=", 0),
    ({"type": "linear", "terms": {"x": "50%", "y": "10%"}, "limit": "30%", "direction": "at most", "operator": LE},
     {"x": 0.2, "y": -0.2}, "<=", 0),  # percentage cap on a weighted sum = weighted average
])
def test_constraint_types(constraint, coefs, op, rhs):
    (c,) = parse(record([constraint]))["constraints"]
    assert c["op"] == op and c["rhs"] == pytest.approx(rhs)
    assert {k: v for k, v in c["coefs"].items() if v} == pytest.approx(coefs)


@pytest.mark.parametrize("direction, sense", [
    ("maximize", "maximize"), ("minimum", "minimize"), ("reduce", "minimize"), ("lowest", "minimize"),
    ("most", "maximize"), ("Minimize", "minimize"),
])
def test_objective_sense(direction, sense):
    assert parse(record([], direction=direction))["sense"] == sense


def test_objective_from_vars_list_is_a_sum():
    assert parse(record([], obj_vars=["x", "y"]))["objective"] == {"x": 1.0, "y": 1.0}


def test_linear_without_terms_is_incomplete():
    with pytest.raises(IncompleteFormulation):
        parse(record([{"type": "linear", "terms": {}, "limit": "90", "direction": "at least", "operator": GE}]))


def test_repeated_var_names_are_dropped():
    rec = record([], vars_=("freight", "air", "air"))
    rec["order_mapping"] = {"freight": 0, "air": 1}
    rec["obj_declaration"]["terms"] = {"freight": "1", "air": "1"}
    assert parse(rec)["vars"] == ["freight", "air"]


@pytest.mark.parametrize("document, vars_, integer", [
    ("A bakery makes cakes and pies. Each cake needs 2 eggs.", ("cakes", "pies"), ["cakes", "pies"]),
    ("A farmer has 100 acres to grow wheat and corn. Profit per acre of wheat is $5.", ("wheat", "corn"), []),
    ("He wants to invest in both the wood and bamboo industries.", ("wood", "bamboo industries"), []),
    ("A store can spend at most $1000 on lemons and bananas.", ("lemons", "bananas"), ["lemons", "bananas"]),
    ("Each trip takes 2 hours. Boats and canoes are available.", ("boats", "canoes"), ["boats", "canoes"]),
    ("Mix kg of fertilizer A with the bags of compost.", ("fertilizer A", "compost"), ["fertilizer A", "compost"]),
])
def test_integer_rule(document, vars_, integer):
    """Last case: cues split the variables (fertilizer A continuous, compost integer) -> all integer."""
    assert parse(record([], obj_terms={v: "1" for v in vars_}, document=document, vars_=vars_))["integer"] == integer


def test_identifiers_are_valid_and_unique():
    ids = identifiers(["Feed A", "feed a", "2nd item", "class", "prob", "Y."])
    assert len(set(ids.values())) == 6
    assert all(i.isidentifier() for i in ids.values())
    assert ids["class"] != "class" and ids["prob"] != "prob" and ids["2nd item"].startswith("x_")


def test_rendered_program_matches_in_process_solve():
    rec = record([
        {"type": "upperbound", "var": "x", "limit": "4", "direction": "at most", "operator": LE},
        {"type": "linear", "terms": {"y": "2"}, "limit": "12", "direction": "at most", "operator": LE},
        {"type": "linear", "terms": {"x": "3", "y": "2"}, "limit": "18", "direction": "at most", "operator": LE},
    ])
    model = parse(rec)
    assert solve(model) == ("Optimal", pytest.approx(36))
    r = run_program(to_program(model))
    assert r.status == "Optimal" and r.objective == pytest.approx(36)


@pytest.mark.parametrize("constraint", [
    {"type": "ratio", "var": "x", "limit": "30%", "direction": "at most", "operator": LE},
    {"type": "ratio", "var": "y", "limit": "40%", "direction": "at least", "operator": GE},
    {"type": "linear", "terms": {"x": "0.5", "y": "0.1"}, "limit": "30%", "direction": "at most", "operator": LE},
])
def test_literal_ratio_style_is_the_same_constraint(constraint):
    rec = record([
        {"type": "linear", "terms": {"x": "3", "y": "2"}, "limit": "18", "direction": "at most", "operator": LE},
        constraint,
    ])
    model = parse(rec)
    simplified, literal = to_program(model), to_program(model, ratio_style="literal")
    assert "* (x + y)" in literal and "* (x + y)" not in simplified
    a, b = run_program(simplified), run_program(literal)
    assert a.status == b.status == "Optimal"
    assert a.objective == pytest.approx(b.objective) == pytest.approx(solve(model)[1])
