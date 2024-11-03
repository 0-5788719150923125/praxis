#!/bin/sh

# peer 1
python run.py --host_name api.src.eco --device cpu --batch_size 1 --expert_type glu --port 2101 --shuffle --dense --dev --hivemind --no_dashboard

# peer 2
python run.py --host_name api.src.eco --device cpu --batch_size 1 --expert_type glu --port 2102 --shuffle --dense --dev --hivemind --no_dashboard