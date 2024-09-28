import time
from typing import List

import bittensor as bt
import numpy as np

from folding.protocol import JobSubmissionSynapse
from folding.validators.protein import Protein


def get_energies(
    protein: Protein, responses: List[JobSubmissionSynapse], uids: List[int]
):
    """Takes all the data from reponse synapses, applies the reward pipeline, and aggregates the rewards
    into a single torch.FloatTensor. Also aggregates the RMSDs for logging.

    Returns:
        tuple:
            torch.FloatTensor: A tensor of rewards for each miner.
            torch.FloatTensor: A tensor of RMSDs for each miner.
    """
    event = {}
    event["is_valid"] = [False] * len(uids)
    event["checked_energy"] = [0] * len(uids)
    event["reported_energy"] = [0] * len(uids)
    event["miner_energy"] = [0] * len(uids)
    event["rmsds"] = [0] * len(uids)
    event["process_md_output_time"] = [0] * len(uids)
    event["is_run_valid"] = [0] * len(uids)

    energies = np.zeros(len(uids))

    for i, (uid, resp) in enumerate(zip(uids, responses)):
        # Ensures that the md_outputs from the miners are parsed correctly
        try:
            start_time = time.time()
            can_process = protein.process_md_output(
                md_output=resp.md_output,
                hotkey=resp.axon.hotkey,
                state=resp.miner_state,
                seed=resp.miner_seed,
            )
            event["process_md_output_time"][i] = time.time() - start_time

            if not can_process:
                continue

            if resp.dendrite.status_code != 200:
                bt.logging.info(
                    f"uid {uid} responded with status code {resp.dendrite.status_code}"
                )
                continue

            energy = protein.get_energy()
            rmsd = protein.get_rmsd()

            if energy == 0:
                continue

            start_time = time.time()
            is_valid, checked_energy, miner_energy = protein.is_run_valid()
            event["is_run_valid"][i] = time.time() - start_time

            energies[i] = energy if is_valid else 0

            event["checked_energy"][i] = checked_energy
            event["miner_energy"][i] = miner_energy
            event["is_valid"][i] = is_valid
            event["reported_energy"][i] = float(energy)
            event["rmsds"][i] = float(rmsd)

        except Exception as E:
            # If any of the above methods have an error, we will catch here.
            bt.logging.error(
                f"Failed to parse miner data for uid {uid} with error: {E}"
            )
            continue

    return energies, event
