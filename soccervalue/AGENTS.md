# AGENTS.md

Pass-success models. Recording a finished round and pushing it are root rules in `AGENTS.md`.

## Always

- Model B, logistic regression on pass geometry plus a few hand-built spatial features, is the baseline from the first round. The native-end check in `results/trusted-subset/conclusion.json` did not confirm that gain. Do not deepen the raster CNN on the strength of the first round.
- The match split is fixed in `soccervalue/matches.py`. Do not retune it from test scores.
- `python -m soccervalue.run_pass_success` writes local artifacts under `outputs/pass_success/`. Copy the official conclusion to `results/pass-success/conclusion.json` before committing the round.
