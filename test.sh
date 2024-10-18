#!/bin/sh

# peer 1
python run.py --host_name api.src.eco --device cpu --batch_size 1 --expert_type peer --shuffle --dense --reset --dev --hivemind --no_dashboard

# peer 2
python run.py --host_name api.src.eco --device cpu --batch_size 1 --expert_type peer --port 2101 --shuffle --dense --dev --hivemind --no_dashboard