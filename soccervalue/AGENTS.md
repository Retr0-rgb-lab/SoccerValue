# AGENTS.md

Pass-success models. Recording a finished round and pushing it are root rules in `AGENTS.md`.

## Always

- Model B, logistic regression on pass geometry plus a few hand-built spatial features, is the baseline to beat. Do not deepen the raster CNN until a checked subset of kick frames and end points still shows that the raster beats B.
- The match split is fixed in `soccervalue/matches.py`. Do not retune it from test scores.
- `python -m soccervalue.run_pass_success` writes local artifacts under `outputs/pass_success/`. Copy the official conclusion to `results/pass-success/conclusion.json` before committing the round.
