# AGENTS.md

SoccerValue estimates the recorded outcome of a ground pass from tracking. Each finished experiment round keeps its official conclusion in git.

## Always

- After each experiment round, write the official conclusions and the numbers that support them to `results/<round>/conclusion.json`. `outputs/` is local and gitignored. It is not the published record.
- Commit that record with the code that produced it, then push to `origin`. A round is unfinished while the conclusion exists only on this machine.
- Do not recommend unexecuted passes from the current success models. The recorded end point is often the next event, not the intended target.

## Commands

- Test: `python -m unittest tests.test_pass_success`
- Pass-success round: `python -m soccervalue.run_pass_success`

## Index

- `soccervalue/AGENTS.md` — when changing or rerunning the pass-success models
