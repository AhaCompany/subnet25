#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Execute the Python script
python3 ./neurons/validator.py \
    --netuid 25 \
    --subtensor.network finney \
    --wallet.name <test_coldkey> \
    --wallet.hotkey <test_hotkey> \
    --axon.port <your_port> \
    --neuron.queue_size <number of pdb_ids to submit> \
    --neuron.sample_size <number of miners per pdb_id> \