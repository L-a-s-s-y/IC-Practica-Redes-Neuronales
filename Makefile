# define the name of the virtual environment directory
VENV := tf

# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	python -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# venv is a shortcut target
venv: $(VENV)/bin/activate

run: venv
	./$(VENV)/bin/python main.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: all venv run clean
