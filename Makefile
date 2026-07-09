.PHONY: runner test test-mac slice suite-local leaderboard demo thumbnail clean

runner:
	swift build -c release --package-path runner

test:
	. .venv/bin/activate && pytest tests/test_verify_score.py tests/test_llm.py -v

test-mac: runner
	. .venv/bin/activate && pytest tests/ -v

# Phase 0 exit criterion: golden kernel passes end to end
slice: runner
	python scripts/run_problem.py p001_vector_add tests/golden_kernels/vector_add.metal

# Free, fully-local sweep (requires `ollama pull qwen2.5-coder:14b`)
suite-local:
	python scripts/run_suite.py --provider ollama --model qwen2.5-coder:14b --mode repair --k 5

leaderboard:
	python scripts/make_leaderboard.py

# Regenerate the standalone HTML leaderboard demo from results/raw/*.json
demo:
	python scripts/make_demo.py

# Regenerate portfolio thumbnail (SVG + PNG) from results/raw/*.json
thumbnail:
	python scripts/make_thumbnail.py

clean:
	rm -rf runner/.build results/raw/*.json
