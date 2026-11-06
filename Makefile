.PHONY: venv deps run

venv:
	python -m venv .venv
	.venv/bin/pip install -U pip

deps: venv
	.venv/bin/pip install -r requirements.txt

# Usage: make run MESH=path/to/file.stl ARGS="--res 100 --U 1.0"
run: deps
	@if [ -z "$(MESH)" ]; then \
		echo "Usage: make run MESH=path/to/file.stl [ARGS='--res 100 --U 1.0']"; exit 1; \
	fi
	bash run_demo.sh $(MESH) $(ARGS)
