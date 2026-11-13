PROFILE ?= consumer24
MODEL ?= fast
UV ?= uv run

FORBIDDEN_LICENCES = GPL;AGPL;LGPL;GNU General Public License;GNU Affero General Public License;GNU Lesser General Public License;GNU Library;Server Side Public License;SSPL

.PHONY: setup check lint typecheck test licences gpu-test up down

setup:
	uv sync
	$(UV) pre-commit install

check: lint typecheck test licences

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

typecheck:
	$(UV) mypy

test:
	$(UV) pytest -q -m "not gpu and not slow"

licences:
	$(UV) pip-licenses --partial-match --fail-on="$(FORBIDDEN_LICENCES)" --format=plain > /dev/null
	$(UV) python scripts/check_forbidden.py
	@echo "licence gate OK"

gpu-test:
	$(UV) pytest -q -m gpu

up:
	$(UV) python -m docparse.vlm.serve $(PROFILE) $(MODEL) > docker/.env.vllm
	docker compose --env-file docker/.env.vllm -f docker/compose.yaml up -d vllm

down:
	docker compose --env-file docker/.env.vllm -f docker/compose.yaml down
