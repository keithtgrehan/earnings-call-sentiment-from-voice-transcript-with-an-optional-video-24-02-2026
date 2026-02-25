PYTHON := python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
CLI := $(VENV)/bin/earnings-call-sentiment

SMOKE_URL ?= https://www.youtube.com/watch?v=BaW_jenozKc
SMOKE_OUT := ./_smoke_out
SMOKE_CACHE := ./_smoke_cache

.PHONY: setup lint smoke clean

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)

setup: $(VENV_PY)
	$(VENV_PIP) install -U pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -e .

lint: setup
	$(VENV_PIP) install ruff
	$(RUFF) check src
	$(VENV_PY) -m py_compile $$(find src -type f -name "*.py")

smoke: setup
	mkdir -p $(SMOKE_OUT) $(SMOKE_CACHE)
	$(CLI) \
		--youtube-url '$(SMOKE_URL)' \
		--cache-dir $(SMOKE_CACHE) \
		--out-dir $(SMOKE_OUT)
	@echo
	@echo "Smoke artifacts:"
	@ls -la $(SMOKE_OUT) $(SMOKE_CACHE)

clean:
	rm -rf ./_smoke_out ./_smoke_cache build dist
