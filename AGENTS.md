# AGENTS.md

SoccerValue estimates the recorded outcome of a ground pass from tracking. Each finished experiment round keeps its official conclusion in git.

## Always

- After each experiment round, write the official conclusions and the numbers that support them to `results/<round>/conclusion.json`. `outputs/` is local and gitignored. It is not the published record.
- Commit that record with the code that produced it, then push to `origin`. A round is unfinished while the conclusion exists only on this machine.
- Do not recommend unexecuted passes from the current success models. The endpoint is now the teammate the passer aimed at, read from the kick, so it no longer carries the result. That is necessary but not sufficient: the model still conditions on the geometry and execution of a pass that was actually played.

## Commands

- Test: `python -m unittest tests.test_pass_success tests.test_pass_intent tests.test_state_value tests.test_world_ball`
- Pass-success round: `python -m soccervalue.run_pass_success`
- Intent-endpoint round: `python -m soccervalue.run_pass_intent`
- Intent-model round: `python -m soccervalue.run_pass_intent_model`
- Intent-endpoint round: `python -m soccervalue.run_pass_intent_endpoint`
- Pass-value round: `python -m soccervalue.run_pass_value`
- Value-surface round: `python -m soccervalue.run_value_surface`
- Symmetric-value round: `python -m soccervalue.run_value_symmetric`

## Index

- `soccervalue/AGENTS.md` — when changing or rerunning the pass-success models
