.PHONY: runner test test-mac slice suite-local leaderboard clean

runner:
	swift build -c release --package-path runner

test:
	pytest tests/test_verify_score.py -v

test-mac: runner
	pytest tests/ -v

# Phase 0 exit criterion: golden kernel passes end to end
slice: runner
	python scripts/run_problem.py p001_vector_add tests/golden_kernels/vector_add.metal

# Free, fully-local sweep (requires `ollama pull qwen2.5-coder:14b`)
suite-local:
	python scripts/run_suite.py --provider ollama --model qwen2.5-coder:14b --mode repair --k 5

leaderboard:
	python scripts/make_leaderboard.py

clean:
	rm -rf runner/.build results/raw/*.json
