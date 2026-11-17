# Section 2.4 results

Tables and figures written by `refine_experiments.py` with `RUN_EXPERIMENTS = True`.
All games are played with the group's `environment.sim_game` against `default_opponent.py`.

Start here:
- `python refine_experiments.py --evaluate` checks the submitted agent (`rlagent/agentE.py`) in seconds.
- `python refine_experiments.py` retrains the agent (about 4 minutes on one CPU core).
- `pytest test_refine.py` runs the checks on the 2.4 code (about 2 seconds).
