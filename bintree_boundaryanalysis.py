
from typing import Callable, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm

from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights

from random import randrange
from numpy.polynomial.laguerre import lagval

from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal
from rl.gen_utils.plot_funcs import plot_list_of_curves

from pprint import pprint

TrainingDataType = Tuple[int, float, float]


def continuation_curve(
    func: FunctionApprox[Tuple[float, float]],
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return func.evaluate([(t, p) for p in prices])


def exercise_curve(
    strike: float,
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return np.array([max(strike - p, 0) for p in prices])


def put_option_exercise_boundary(
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    num_steps: int,
    strike: float
) -> Tuple[Sequence[float], Sequence[float]]:
    x: List[float] = []
    y: List[float] = []
    prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
    for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            y.append(max(ll))
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    return x, y



spot_price_val: float = 36.0
strike_val: float = 40.0
expiry_val: float = 1.0
rate_val: float = 0.06
vol_val: float = 0.2
num_scoring_paths: int = 100
num_steps_scoring: int = 50

num_steps_lspi: int = 10
num_training_paths_lspi: int = 1000
spot_price_frac_lspi: float = 0.1
training_iters_lspi: int = 8

opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=lambda _, x: max(strike_val - x, 0),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=100
    )

vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
bin_tree_price: float = vf_seq[0][NonTerminal(0)]
bin_tree_ex_boundary: Sequence[Tuple[float, float]] = \
opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
bin_tree_x, bin_tree_y = zip(*bin_tree_ex_boundary)

bin_tree_x