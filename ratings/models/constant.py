"""Baseline that gives every match the same gap, so it only predicts the overall home/draw/away rates."""


def new_model(name="baseline_home_rate"):
    return {"kind": "constant", "name": name}


def start_season(model, year):
    pass


def gap(model, match):
    return 0.0


def update(model, match):
    pass
