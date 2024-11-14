#!/bin/sh

# peer 1
python run.py --device cpu --batch_size 1 --port 2101 --shuffle --sparse --dev --hivemind --no_dashboard --debug --quiet

# peer 2
python run.py --device cpu --batch_size 1 --port 2102 --shuffle --sparse --dev --hivemind --no_dashboard --debug --quiet