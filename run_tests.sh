#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)" 
pytest tests/modelling/ tests/data/ tests/utils/