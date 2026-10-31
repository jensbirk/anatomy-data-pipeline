# Anatomy of a Data Pipeline: artifact
#
#   make run experiments/<dir>         run every entry of <dir>/manifest.yaml
#
# Positional argument after the target (e.g. "make run experiments/foo").
ARG := $(word 2,$(MAKECMDGOALS))
$(eval $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)):;@:)

.PHONY: help run

help:
	@sed -n '3,4p' Makefile | sed 's/^# //'

run:
	@test -n "$(ARG)" || { echo "Usage: make run experiments/<dir>"; exit 1; }
	uv run -m system.experiment_runner --exp_dir $(ARG)
