UV ?= uv
CONFIG ?= configs/tomato.toml
SEED ?= 0
OUTPUT ?= runs/tefed-seed$(SEED)
RUN = $(UV) run --locked --extra cpu

.PHONY: help install check test smoke methods train docker

help:
	@echo "install   Sync the locked CPU environment"
	@echo "check     Check the lockfile"
	@echo "test      Run the software test suite"
	@echo "smoke     Run the synthetic software fixture"
	@echo "methods   List implemented methods"
	@echo "train     Train with a CPU-ready local CONFIG, SEED, and OUTPUT"
	@echo "docker    Build the optional CPU image (Docker required)"

install:
	$(UV) sync --locked --extra cpu

check:
	$(UV) lock --check

test:
	$(RUN) python -m unittest discover -s tests -t . -v

smoke:
	$(RUN) python -m tefed.main smoke --output runs/software-smoke

methods:
	$(RUN) python -m tefed.main methods

train:
	$(RUN) python -m tefed.main tefed --config "$(CONFIG)" --seed $(SEED) --output "$(OUTPUT)"

docker:
	docker build -t tefed:cpu .
