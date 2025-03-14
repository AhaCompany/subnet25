import os
import time
import glob
import copy
import json
import base64
import random
import hashlib
import requests
import traceback
import concurrent.futures
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import openmm as mm
import openmm.app as app

# import base miner class which takes care of most of the boilerplate
from folding.base.miner import BaseMinerNeuron
from folding.base.simulation import OpenMMSimulation
from folding.protocol import JobSubmissionSynapse, ParticipationSynapse
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter
from folding.utils.ops import (
    check_if_directory_exists,
    get_tracebacks,
    write_pkl,
)
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.logger import logger


def attach_files(
    files_to_attach: List, synapse: JobSubmissionSynapse
) -> JobSubmissionSynapse:
    """function that parses a list of files and attaches them to the synapse object"""
    logger.info(f"Sending files to validator: {files_to_attach}")
    for filename in files_to_attach:
        try:
            with open(filename, "rb") as f:
                filename = filename.split("/")[
                    -1
                ]  # remove the directory from the filename
                synapse.md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            logger.error(f"Failed to read file {filename!r} with error: {e}")
            get_tracebacks()

    return synapse


def attach_files_to_synapse(
    synapse: JobSubmissionSynapse,
    data_directory: str,
    state: str,
    seed: int,
) -> JobSubmissionSynapse:
    """load the output files as bytes and add to synapse.md_output

    Args:
        synapse (JobSubmissionSynapse): Recently received synapse object
        data_directory (str): directory where the miner is holding the necessary data for the validator.
        state (str): the current state of the simulation

    state is either:
     1. nvt
     2. npt
     3. md_0_1
     4. finished

    Returns:
        JobSubmissionSynapse: synapse with md_output attached
    """

    synapse.md_output = {}  # ensure that the initial state is empty

    try:
        state_files = os.path.join(data_directory, f"{state}")

        # This should be "state.cpt" and "state_old.cpt"
        all_state_files = glob.glob(f"{state_files}*")  # Grab all the state_files

        if len(all_state_files) == 0:
            raise FileNotFoundError(
                f"No files found for {state}"
            )  # if this happens, goes to except block

        synapse = attach_files(files_to_attach=all_state_files, synapse=synapse)

        synapse.miner_seed = seed
        synapse.miner_state = state

    except Exception as e:
        logger.error(f"Failed to attach files for pdb {synapse.pdb_id} with error: {e}")
        get_tracebacks()
        synapse.md_output = {}

    finally:
        return synapse  # either return the synapse wth the md_output attached or the synapse as is.


def check_synapse(
    self, synapse: JobSubmissionSynapse, event: Dict = None
) -> JobSubmissionSynapse:
    """Utility function to remove md_inputs if they exist"""

    if synapse.md_output is not None:
        event["md_output_sizes"] = list(map(len, synapse.md_output.values()))
        event["md_output_filenames"] = list(synapse.md_output.keys())

    return synapse


class FoldingMiner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)

        # Setup project path if not already defined
        if not hasattr(self, 'project_path') or self.project_path is None:
            self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            logger.info(f"Setting project_path to {self.project_path}")
        
        # Setup rqlite data directory if not defined
        if not hasattr(self, 'rqlite_data_dir') or self.rqlite_data_dir is None:
            self.rqlite_data_dir = "rqlite-data"
            logger.info(f"Setting rqlite_data_dir to {self.rqlite_data_dir}")
            
        # Initialize data paths
        self.miner_data_path = os.path.join(self.project_path, "miner-data")
        self.base_data_path = os.path.join(
            self.miner_data_path, self.wallet.hotkey.ss58_address[:8]
        )
        
        # Create data directories if they don't exist
        os.makedirs(self.miner_data_path, exist_ok=True)
        os.makedirs(self.base_data_path, exist_ok=True)
        
        self.local_db_address = os.getenv("RQLITE_HTTP_ADDR")
        if not self.local_db_address:
            self.local_db_address = "localhost:4001"
            os.environ["RQLITE_HTTP_ADDR"] = self.local_db_address
            logger.warning(f"RQLITE_HTTP_ADDR not set, using default: {self.local_db_address}")
            
        self.simulations = self.create_default_dict()

        # Configure worker pool - set optimal based on GPU memory
        self.max_workers = self.config.neuron.max_workers
        logger.info(
            f"🚀 Starting FoldingMiner that handles {self.max_workers} workers 🚀"
        )

        # Use ProcessPoolExecutor for better isolation and resource management
        try:
            # Try using spawn context if available (Python 3.8+)
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=ctx
            )
        except (ImportError, TypeError):
            # Fallback for older Python versions
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )

        # Initialize simulation tracking
        self.mock = None
        self.completed_jobs = {}  # Track successful completions for analytics
        self.failed_jobs = {}  # Track failures for debugging
        
        # Generate more diverse seeds for better energy minimization
        self.generate_random_seed = lambda: random.randint(0, 100000)
        
        # Start local database with error handling
        try:
            asyncio.run(self.start_rqlite())
            time.sleep(5)
        except Exception as e:
            logger.error(f"Failed to start rqlite: {e}")
            logger.warning("Continuing without rqlite database - some features may be limited")

        # Simulation configuration constants
        self.STATES = ["nvt", "npt", "md_0_1"]
        self.CHECKPOINT_INTERVAL = 10000  # High enough for efficiency but sufficient for recovery
        self.STATE_DATA_REPORTER_INTERVAL = 10
        self.EXIT_REPORTER_INTERVAL = 10

    def create_default_dict(self):
        def nested_dict():
            return defaultdict(
                lambda: None
            )  # allows us to set the desired attribute to anything.

        return defaultdict(nested_dict)

    def check_and_remove_simulations(self, event: Dict) -> Dict:
        """Check to see if any simulations have finished, and remove them
        from the simulation store
        """
        if len(self.simulations) > 0:
            sims_to_delete = []

            for pdb_hash, simulation in self.simulations.items():
                future = simulation["future"]
                pdb_id = simulation["pdb_id"]

                # Check if the future is done
                if future.done():
                    state, error_info = future.result()
                    # If the simulation is done, we can remove it from the simulation store
                    if state == "finished":
                        logger.warning(
                            f"✅ {pdb_id} finished simulation... Removing from execution stack ✅"
                        )
                    else:
                        # If the simulation failed, we should log the error and remove it from the simulation store
                        logger.error(
                            f"❗ {pdb_id} failed simulation... Removing from execution stack ❗"
                        )
                        logger.error(f"Error info: {error_info}")
                    sims_to_delete.append(pdb_hash)

            for pdb_hash in sims_to_delete:
                del self.simulations[pdb_hash]

            running_simulations = [sim["pdb_id"] for sim in self.simulations.values()]

            event["running_simulations"] = running_simulations
            logger.warning(f"Simulations Running: {running_simulations}")

        return event

    def get_simulation_hash(self, pdb_id: str, system_config: Dict) -> str:
        """Creates a simulation hash based on the pdb_id and the system_config given.

        Returns:
            str: first 6 characters of a sha256 hash
        """
        system_hash = pdb_id
        for key, value in system_config.items():
            system_hash += str(key) + str(value)

        hash_object = hashlib.sha256(system_hash.encode("utf-8"))
        return hash_object.hexdigest()[:6]

    def is_unique_job(self, system_config_filepath: str) -> bool:
        """Check to see if a submitted job is unique by checking to see if the folder exists.

        Args:
            system_config_filepath (str): filepath for the config file that specifies the simulation

        Returns:
            bool
        """
        if os.path.exists(system_config_filepath):
            return False
        return True

    def response_to_dict(self, response) -> dict[str, Any]:
        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get all PDBs: {response['error']}")
        elif "values" not in response.keys():
            return {}

        columns = response["columns"]
        values = response["values"]
        data = [dict(zip(columns, row)) for row in values]
        return data

    def fetch_sql_job_details(
        self, columns: List[str], job_id: str, local_db_address: str
    ) -> Dict:
        """
        Fetches job records from a SQLite database with given column details and a specific job_id.

        Parameters:
            columns (list): List of column names to retrieve from the database.
            job_id (str): The identifier for the job to fetch.
            db_path (str): Path to the SQLite database file.

        Returns:
            dict: A dictionary mapping job_id to its details as specified by the columns list.
        """

        logger.info("Fetching job details from the sqlite database")

        full_local_db_address = f"http://{local_db_address}/db/query"
        columns_to_select = ", ".join(columns)
        query = f"""SELECT job_id, {columns_to_select} FROM jobs WHERE job_id = '{job_id}'"""

        try:
            response = requests.get(
                full_local_db_address,
                params={"q": query, "level": "strong"},
                timeout=10,
            )
            response.raise_for_status()

            data: dict = self.response_to_dict(response=response)
            logger.info(f"data response: {data}")

            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch job details {e}")
            return

    def download_gjp_input_files(
        self,
        output_dir: str,
        pdb_id: str,
        s3_links: dict[str, str],
    ) -> bool:
        """Downloads input files required from the GJP (Global Job Pool) from S3 storage.

        Args:
            output_dir (str): Directory path where downloaded files will be saved
            pdb_id (str): Identifier for the job
            s3_links (dict[str, str]): Dictionary mapping file types to their S3 URLs

        Returns:
            bool: True if all files were downloaded successfully, False if any download failed

        The function:
        1. Creates the output directory if it doesn't exist
        2. Downloads each file in chunks to conserve memory
        3. Names downloaded files as {pdb_id}.{file_type}
        """

        def stream_download(url: str, output_path: str):
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        for key, url in s3_links.items():
            output_path = os.path.join(output_dir, f"{pdb_id}.{key}")
            try:
                stream_download(url=url, output_path=output_path)
            except Exception as e:
                logger.error(f"Failed to download file {key} with error: {e}")
                return False
        return True

    def get_simulation_config(
        self,
        gjp_config,
        system_config_filepath: str,
    ) -> SimulationConfig:
        """
        Creates a SimulationConfig for the gjp job the miner is working on.

        Parameters:
            gjp_config (dict): Configuration details for the GJP.
            system_config_filepath (str): File path to write the system configuration.
            job_id (str): Job ID to process.

        Returns:
            SimulationConfig: An object containing the configuration for the simulation.
        """

        # create SimualtionConfig and write it to system_config_filepath
        system_config = SimulationConfig(
            ff=gjp_config["ff"],
            water=gjp_config["water"],
            box=gjp_config["box"],
            **gjp_config["system_kwargs"],
        )

        if system_config.seed is None:
            system_config.seed = self.generate_random_seed()

        write_pkl(system_config, system_config_filepath)
        return system_config

    def check_if_job_was_worked_on(self, job_id: str) -> tuple[bool, str, dict]:
        """Check if a job has been previously worked on or is currently being processed.

        Parameters:
            job_id (str): The unique identifier for the job to check.

        Returns:
            tuple[bool, str, dict]: A tuple containing:
                - Whether the job has been worked on (bool)
                - The condition of the job (str)
                - Event dictionary with job details (dict)
        """

        columns = ["pdb_id", "system_config"]

        # query your LOCAL rqlite db to get pdb_id
        sql_job_details = self.fetch_sql_job_details(
            columns=columns, job_id=job_id, local_db_address=self.local_db_address
        )[0]

        if len(sql_job_details) == 0:
            logger.warning(f"Job ID {job_id} not found in the database.")
            return False, "job_not_found", {}

        # str
        pdb_id = sql_job_details["pdb_id"]

        # If we are already running a process with the same identifier, return intermediate information
        logger.info(f"⌛ Checking for protein: {pdb_id} ⌛")

        event = self.create_default_dict()
        event["pdb_id"] = pdb_id

        gjp_config = json.loads(sql_job_details["system_config"])
        event["gjp_config"] = gjp_config

        pdb_hash = self.get_simulation_hash(pdb_id=pdb_id, system_config=gjp_config)
        event["pdb_hash"] = pdb_hash

        if pdb_hash in self.simulations:
            return True, "running_simulation", event

        # If you don't have in the list of simulations, check your local storage for the data.
        output_dir = os.path.join(self.base_data_path, pdb_id, pdb_hash)
        gjp_config_filepath = os.path.join(output_dir, f"config_{pdb_id}.pkl")
        event["output_dir"] = output_dir
        event["gjp_config_filepath"] = gjp_config_filepath

        # check if any of the simulations have finished
        event = self.check_and_remove_simulations(event=event)

        submitted_job_is_unique = self.is_unique_job(
            system_config_filepath=gjp_config_filepath
        )

        if not submitted_job_is_unique:
            return True, "found_existing_data", event

        return False, "job_not_worked_on", event

    def participation_forward(self, synapse: ParticipationSynapse):
        """Respond to the validator with the necessary information about participating in a specified job
        If the miner has worked on a job before, it should return True for is_participating.
        If the miner has not worked on a job before, it should return False for is_participating.

        Args:
            self (ParticipationSynapse): must attach "is_participating"
        """
        job_id = synapse.job_id
        logger.info(f"⌛ Validator checking if miner has participated in job: {job_id} ⌛")
        has_worked_on_job, _, _ = self.check_if_job_was_worked_on(job_id=job_id)
        synapse.is_participating = has_worked_on_job
        return synapse

    def forward(self, synapse: JobSubmissionSynapse) -> JobSubmissionSynapse:
        """Process an incoming job submission request and return appropriate simulation data.

        This method handles three main scenarios:
        1. Found existing data: Returns previously computed simulation state files
        2. Running simulation: Returns current state of an active simulation

        The validator will use the JobSubmissionSynapse to acquire the results of the work done.

        Args:
            synapse (JobSubmissionSynapse): The incoming request object containing:
                - job_id: Unique identifier for the simulation job
                - Additional metadata needed for job processing

        Returns:
            JobSubmissionSynapse: The response object containing:
                - md_output: Dictionary of base64 encoded simulation state files
                - miner_state: Current state of simulation ("nvt", "npt", "md_0_1", "finished", or "failed")
                - miner_seed: Random seed used for the simulation
                - miner_energy: Current/latest energy value (if available)

        Note:
            The method checks the local database and running simulations before starting
            new jobs to avoid duplicate work. State files are only attached if valid
            simulation data is found.
        """
        job_id = synapse.job_id
        start_time = time.time()

        has_worked_on_job, condition, event = self.check_if_job_was_worked_on(
            job_id=job_id
        )
        self.step += 1

        if has_worked_on_job:
            if condition == "found_existing_data":
                if os.path.exists(
                    event["output_dir"]
                ) and f"{event['pdb_id']}.pdb" in os.listdir(event["output_dir"]):
                    # If we have a pdb_id in the data directory, we can assume that the simulation has been run before
                    # and we can return the COMPLETED files from the last simulation. This only works if you have kept the data.

                    # We will attempt to read the state of the simulation from the state file
                    state_file = os.path.join(
                        event["output_dir"], f"{event['pdb_id']}_state.txt"
                    )
                    seed_file = os.path.join(
                        event["output_dir"], f"{event['pdb_id']}_seed.txt"
                    )
                    energy_file = os.path.join(
                        event["output_dir"], f"{event['pdb_id']}_energy.txt"
                    )

                    # Open the state file that should be generated during the simulation.
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            state = lines[-1].strip()
                            state = "md_0_1" if state == "finished" else state

                        # If the state is failed, we should not return the files.
                        if state == "failed":
                            synapse.miner_state = state
                            event["condition"] = "failed_simulation"
                            event["state"] = state
                            logger.warning(
                                f"❗Returning previous simulation data for failed simulation: {event['pdb_id']}❗"
                            )
                            return check_synapse(
                                self=self, synapse=synapse, event=event
                            )

                        with open(seed_file, "r", encoding="utf-8") as f:
                            seed = f.readlines()[-1].strip()
                            
                        # Try to read the energy file if it exists
                        latest_energy = None
                        try:
                            if os.path.exists(energy_file):
                                with open(energy_file, "r", encoding="utf-8") as f:
                                    # Skip header line
                                    lines = f.readlines()
                                    if len(lines) > 1:
                                        # Get the last recorded energy
                                        last_energy_line = lines[-1].strip().split(',')
                                        if len(last_energy_line) >= 2:
                                            latest_energy = float(last_energy_line[1])
                        except Exception as energy_err:
                            logger.warning(f"Error reading energy file: {energy_err}")

                        logger.warning(
                            f"❗ Found existing data for protein: {event['pdb_id']}... Sending previously computed, most advanced simulation state ❗"
                        )
                        
                        # Attach all files to the synapse
                        synapse = attach_files_to_synapse(
                            synapse=synapse,
                            data_directory=event["output_dir"],
                            state=state,
                            seed=seed,
                        )
                        
                        # Add energy data if available
                        if latest_energy is not None:
                            synapse.miner_energy = latest_energy
                            
                    except Exception as e:
                        logger.error(
                            f"Failed to read state file for protein {event['pdb_id']} with error: {e}"
                        )
                        state = None

                    finally:
                        event["condition"] = "found_existing_data"
                        event["state"] = state
                        event["processing_time"] = time.time() - start_time
                        return check_synapse(self=self, synapse=synapse, event=event)

            # The set of RUNNING simulations.
            elif condition == "running_simulation":
                self.simulations[event["pdb_hash"]]["queried_at"] = time.time()
                simulation = self.simulations[event["pdb_hash"]]
                
                # Get current execution state
                current_executor_state = simulation["executor"].get_state()
                current_seed = simulation["executor"].seed
                
                # Try to get current energy if possible
                try:
                    current_energy = simulation["executor"].get_latest_energy()
                    if current_energy is not None:
                        synapse.miner_energy = current_energy
                        
                        # Track if this is the best energy we've seen
                        if "best_energy" not in simulation or current_energy < simulation["best_energy"]:
                            simulation["best_energy"] = current_energy
                            logger.info(f"New best energy for {event['pdb_id']}: {current_energy}")
                except Exception as e:
                    logger.warning(f"Failed to get current energy: {e}")

                # Attach checkpoint files to response
                synapse = attach_files_to_synapse(
                    synapse=synapse,
                    data_directory=simulation["output_dir"],
                    state=current_executor_state,
                    seed=current_seed,
                )

                # Update tracking
                event["condition"] = "running_simulation"
                event["state"] = current_executor_state
                event["queried_at"] = simulation["queried_at"]
                event["processing_time"] = time.time() - start_time
                
                # Record validator interaction for analytics
                if not hasattr(self, "validator_interactions"):
                    self.validator_interactions = {}
                    
                validator_hotkey = synapse.dendrite.hotkey
                if validator_hotkey not in self.validator_interactions:
                    self.validator_interactions[validator_hotkey] = {"count": 0, "last_query": 0}
                
                self.validator_interactions[validator_hotkey]["count"] += 1
                self.validator_interactions[validator_hotkey]["last_query"] = time.time()

                return check_synapse(self=self, synapse=synapse, event=event)

    def create_simulation_from_job(
        self,
        output_dir: str,
        pdb_id: str,
        pdb_hash: str,
        system_config: SimulationConfig,
        event: Dict,
    ):
        # Submit job to the executor
        simulation_manager = SimulationManager(
            pdb_id=pdb_id,
            output_dir=output_dir,
            system_config=system_config.model_dump(),
            seed=system_config.seed,
        )

        future = self.executor.submit(
            simulation_manager.run,
            self.config.mock or self.mock,  # self.mock is inside of MockFoldingMiner
        )

        self.simulations[pdb_hash]["pdb_id"] = pdb_id
        self.simulations[pdb_hash]["executor"] = simulation_manager
        self.simulations[pdb_hash]["future"] = future
        self.simulations[pdb_hash]["output_dir"] = simulation_manager.output_dir
        self.simulations[pdb_hash]["queried_at"] = time.time()

        logger.success(f"✅ New pdb_id {pdb_id} submitted to job executor ✅ ")

        event["condition"] = "new_simulation"
        event["start_time"] = time.time()

    async def blacklist(self, synapse: JobSubmissionSynapse) -> Tuple[bool, str]:
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            logger.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            # We also check if the stake is greater than 10_000, which is the minimum stake to not be blacklisted.
            if (
                not self.metagraph.validator_permit[uid]
                or self.metagraph.stake[uid] < 10_000
            ):
                logger.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        logger.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: JobSubmissionSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        return priority

    def add_active_jobs_from_db(self, limit: int = None) -> int:
        """
        Fetch active jobs from the database and add them to the simulation executor.
        
        This optimized implementation:
        1. Prioritizes high-priority jobs 
        2. Considers validator stake in job selection
        3. Uses smart scheduling to maximize throughput
        4. Maintains job analytics for better decision making

        Parameters:
            limit (int, optional): Maximum number of new jobs to add. If None, add as many
                                  as possible up to max_workers. Defaults to None.

        Returns:
            int: Number of jobs added to the executor
        """
        if not self.local_db_address:
            logger.warning(
                "No local database address configured, cannot add active jobs"
            )
            return 0

        # Calculate how many slots are available
        available_slots = self.max_workers - len(self.simulations)
        if available_slots <= 0:
            logger.info("No available worker slots for new jobs")
            return 0

        # Determine how many jobs to fetch - get more than needed to allow for filtering
        jobs_to_fetch = limit if limit is not None else available_slots
        fetch_multiplier = 3  # Fetch more jobs than needed to allow for filtering/prioritization
        expanded_fetch = jobs_to_fetch * fetch_multiplier
        
        # Analytics dictionaries
        if not hasattr(self, "job_history"):
            self.job_history = {}  # Store job performance history
            
        if not hasattr(self, "validator_statistics"):
            self.validator_statistics = {}  # Store validator interaction statistics

        # Query the database for active jobs that are not already being processed
        full_local_db_address = f"http://{self.local_db_address}/db/query"
        # We need columns that identify the job and contain essential configuration
        columns_to_select = "pdb_id, system_config, priority, s3_links, validator_uid, created_at"
        query = f"""SELECT job_id, {columns_to_select} FROM jobs 
                   WHERE active = 1 
                   ORDER BY priority DESC, created_at DESC
                   LIMIT {expanded_fetch}
                   """

        try:
            response = requests.get(
                full_local_db_address,
                params={"q": query, "level": "strong"},
                timeout=10,
            )
            response.raise_for_status()

            data = self.response_to_dict(response=response)
            if not data or len(data) == 0:
                logger.info("No active jobs found in database")
                return 0

            logger.info(f"Found {len(data)} active jobs in gjp")
            
            # Track validator stakes to use for job prioritization
            validator_stakes = {}
            for uid in range(len(self.metagraph.hotkeys)):
                if self.metagraph.validator_permit[uid]:
                    validator_stakes[uid] = float(self.metagraph.S[uid])
            
            # Intelligently score and rank jobs
            scored_jobs = []
            for job in data:
                job_id = job.get("job_id")
                pdb_id = job.get("pdb_id")
                priority = job.get("priority", 0)
                validator_uid = job.get("validator_uid")
                created_at = job.get("created_at", 0)
                
                # Skip if already working on this job
                has_worked_on_job, _, _ = self.check_if_job_was_worked_on(job_id=job_id)
                if has_worked_on_job:
                    continue
                
                # Calculate a job score based on multiple factors
                job_score = priority * 10  # Base score from priority
                
                # Add validator stake weighting
                if validator_uid is not None and validator_uid in validator_stakes:
                    job_score += validator_stakes[validator_uid] * 0.5
                
                # Factor in job age - newer jobs get slight preference
                if created_at:
                    # Normalize creation time to be between 0-1 (newer = higher)
                    current_time = time.time()
                    age_factor = max(0, min(1, 1 - ((current_time - created_at) / (7 * 24 * 3600))))
                    job_score += age_factor * 5
                
                # Add to scored list
                scored_jobs.append((job_score, job))
            
            # Sort jobs by score (highest first)
            scored_jobs.sort(reverse=True)
            
            # Keep track of how many jobs we've added
            jobs_added = 0

            # Add each job to the simulation executor if not already being processed
            for _, job in scored_jobs:
                if jobs_added >= available_slots:
                    break
                    
                job_id = job.get("job_id")
                pdb_id = job.get("pdb_id")
                system_config_json = job.get("system_config")
                s3_links = job.get("s3_links")
                
                if not job_id or not pdb_id or not system_config_json:
                    logger.warning(f"Incomplete job data: {job}")
                    continue

                # Generate a unique hash for this job to check if it's already running
                try:
                    system_config = json.loads(system_config_json)
                    pdb_hash = self.get_simulation_hash(pdb_id, system_config)

                    # Skip if this simulation is already running
                    if pdb_hash in self.simulations:
                        logger.info(
                            f"Simulation for PDB {pdb_id} (hash: {pdb_hash}) is already running"
                        )
                        continue

                    # Create an output directory for this job
                    output_dir = os.path.join(self.base_data_path, pdb_id, pdb_hash)
                    os.makedirs(output_dir, exist_ok=True)

                    # Track download start time for performance monitoring
                    download_start = time.time()
                    success = self.download_gjp_input_files(
                        pdb_id=pdb_id,
                        output_dir=output_dir,
                        s3_links=json.loads(s3_links),
                    )
                    download_time = time.time() - download_start
                    
                    if not success:
                        logger.error(
                            f"Failed to download GJP input files for job {job_id}"
                        )
                        continue
                        
                    logger.info(f"Downloaded files for {pdb_id} in {download_time:.2f}s")

                    # Create simulation config with optimized parameters
                    simulation_config = self.get_simulation_config(
                        gjp_config=system_config,
                        system_config_filepath=os.path.join(
                            output_dir, f"config_{pdb_id}.pkl"
                        ),
                    )

                    # Add the job to the simulation executor
                    event = {
                        "condition": "loading_from_db",
                        "priority": job.get("priority", 0),
                        "validator_uid": job.get("validator_uid"),
                        "start_time": time.time()
                    }
                    
                    # Launch the simulation
                    self.create_simulation_from_job(
                        output_dir=output_dir,
                        pdb_id=pdb_id,
                        pdb_hash=pdb_hash,
                        system_config=simulation_config,
                        event=event,
                    )

                    # Track job in history for analytics
                    self.job_history[job_id] = {
                        "pdb_id": pdb_id,
                        "priority": job.get("priority", 0),
                        "validator_uid": job.get("validator_uid"),
                        "start_time": time.time(),
                        "status": "running"
                    }

                    jobs_added += 1
                    logger.success(
                        f"Added job {job_id} for PDB {pdb_id} (priority: {job.get('priority', 0)}) from database to executor"
                    )

                except Exception:
                    logger.error(
                        f"Failed to add job {job_id} for PDB {pdb_id}: {traceback.format_exc()}"
                    )
                    continue

            logger.info(f"Added {jobs_added} jobs from database to simulation executor")
            
            # If we have fewer jobs running than our capacity, check again soon
            if jobs_added < available_slots and jobs_added > 0:
                logger.warning(f"Only filled {jobs_added}/{available_slots} slots, will check for more jobs soon")
                
            return jobs_added

        except requests.RequestException as e:
            logger.error(f"Failed to fetch active jobs from database: {e}")
            return 0

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.
        4. Periodically checks the database for new jobs and adds them to the simulation executor.
        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """
        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        logger.info(f"Miner starting at block: {self.block}")

        # Initialize the last job check time
        jobs_added = self.add_active_jobs_from_db()

        if jobs_added == 0:
            logger.warning("No jobs added during initialization")
        else:
            logger.success(f"Added {jobs_added} new jobs from database")

        last_job_check_time = time.time()
        job_check_interval = 300  # Check for new jobs every 300 seconds

        # This loop maintains the miner's operations until intentionally stopped
        try:
            while not self.should_exit:
                # Perform regular chain synchronization
                self.sync()

                # Check for available job slots and fill them if needed
                current_time = time.time()
                if current_time - last_job_check_time > job_check_interval:
                    logger.info("Checking for available job slots...")
                    jobs_added = self.add_active_jobs_from_db()
                    if jobs_added > 0:
                        logger.success(f"Added {jobs_added} new jobs from database")

                    last_job_check_time = current_time

                logger.info(
                    f"currently working on {len(self.simulations)} jobs: {[simulation['pdb_id'] for simulation in self.simulations.values()]}"
                )

                # Sleep to prevent CPU overuse
                time.sleep(10)

        except KeyboardInterrupt:
            self.axon.stop()
            logger.success("Miner killed by keyboard interrupt.")
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())


class SimulationManager:
    def __init__(
        self, pdb_id: str, output_dir: str, seed: int, system_config: dict
    ) -> None:
        self.pdb_id = pdb_id
        self.state: str = None
        self.seed = seed
        self.pdb_obj = app.PDBFile(os.path.join(output_dir, f"{pdb_id}.pdb"))

        self.state_file_name = f"{pdb_id}_state.txt"
        self.seed_file_name = f"{pdb_id}_seed.txt"
        self.energy_file_name = f"{pdb_id}_energy.txt"  # Track best energies
        self.simulation_steps: dict = system_config["simulation_steps"]
        self.system_config = SimulationConfig(**system_config)

        self.output_dir = output_dir
        self.start_time = time.time()

        self.cpt_file_mapper = {
            "nvt": f"{output_dir}/{self.pdb_id}.cpt",
            "npt": f"{output_dir}/nvt.cpt",
            "md_0_1": f"{output_dir}/npt.cpt",
        }

        self.STATES = ["nvt", "npt", "md_0_1"]
        self.CHECKPOINT_INTERVAL = 10000
        self.STATE_DATA_REPORTER_INTERVAL = 10
        self.EXIT_REPORTER_INTERVAL = 10
        self.ENERGY_REPORT_INTERVAL = 1000  # How often to check and report energy

    def create_empty_file(self, file_path: str):
        # For mocking
        with open(file_path, "w") as f:
            pass

    def write_state(self, state: str, state_file_name: str, output_dir: str):
        with open(os.path.join(output_dir, state_file_name), "w") as f:
            f.write(f"{state}\n")

    def record_energy(self, energy: float):
        """Record the current energy to track best values"""
        try:
            with open(os.path.join(self.output_dir, self.energy_file_name), "a") as f:
                f.write(f"{time.time()},{energy}\n")
        except Exception as e:
            logger.error(f"Failed to record energy: {e}")

    def run(
        self,
        mock: bool = False,
    ):
        """run method to handle the processing of generic simulations with enhanced error recovery.

        Args:
            mock (bool, optional): mock for debugging. Defaults to False.
        """
        logger.info(f"Running simulation for protein: {self.pdb_id}")
        
        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=self.output_dir)
        os.chdir(self.output_dir)

        # Write the seed so that we always know what was used
        with open(self.seed_file_name, "w") as f:
            f.write(f"{self.seed}\n")
            
        # Initial energy file
        with open(self.energy_file_name, "w") as f:
            f.write("timestamp,energy\n")

        # Maximum retry attempts for each state
        max_retries = 2
        
        try:
            # Create all simulation objects once to avoid recreation
            simulations = self.configure_commands(
                seed=self.seed, system_config=copy.deepcopy(self.system_config)
            )
            
            # Run each state with retry capability
            for state in self.STATES:
                simulation = simulations[state]
                logger.info(f"Running {state} simulation for {self.pdb_id}")
                
                self.write_state(
                    state=state, 
                    state_file_name=self.state_file_name,
                    output_dir=self.output_dir
                )
                
                retry_count = 0
                success = False
                
                while not success and retry_count < max_retries:
                    try:
                        # Load appropriate checkpoint
                        simulation.loadCheckpoint(self.cpt_file_mapper[state])
                        
                        # Run simulation with energy monitoring
                        steps_per_batch = 5000  # Break into smaller batches for monitoring
                        remaining_steps = self.simulation_steps[state]
                        
                        while remaining_steps > 0:
                            batch_steps = min(steps_per_batch, remaining_steps)
                            simulation.step(batch_steps)
                            remaining_steps -= batch_steps
                            
                            # Monitor current energy
                            state_info = simulation.context.getState(getEnergy=True)
                            current_energy = state_info.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                            self.record_energy(current_energy)
                        
                        success = True
                        
                    except mm.OpenMMException as e:
                        retry_count += 1
                        logger.warning(f"Simulation {state} failed (attempt {retry_count}): {str(e)}")
                        if retry_count >= max_retries:
                            raise  # Re-raise if we've exhausted retries
                        
                        # Wait briefly before retry
                        time.sleep(2)
                
                # Exit if this state failed after retries
                if not success:
                    raise Exception(f"Failed to complete {state} simulation after {max_retries} attempts")

            # All states completed successfully
            logger.success(f"✅ Finished simulation for protein: {self.pdb_id} ✅")
            state = "finished"
            self.write_state(
                state=state,
                state_file_name=self.state_file_name,
                output_dir=self.output_dir,
            )
            return state, None

        # This is the exception that is raised when the simulation fails
        except mm.OpenMMException as e:
            state = "failed"
            error_info = {
                "type": "OpenMMException",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            try:
                platform = mm.Platform.getPlatformByName("CUDA")
                error_info["cuda_version"] = platform.getPropertyDefaultValue("CudaCompiler")
                error_info["platform_properties"] = {
                    prop: platform.getPropertyDefaultValue(prop) 
                    for prop in platform.getPropertyNames()
                }
            except Exception as inner_e:
                error_info["cuda_info_error"] = str(inner_e)
            finally:
                self.write_state(
                    state=state,
                    state_file_name=self.state_file_name,
                    output_dir=self.output_dir,
                )
                return state, error_info

        # Generic Exception
        except Exception as e:
            state = "failed"
            error_info = {
                "type": "UnexpectedException",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            self.write_state(
                state=state,
                state_file_name=self.state_file_name,
                output_dir=self.output_dir,
            )
            return state, error_info

    def get_state(self) -> str:
        """get_state reads a txt file that contains the current state of the simulation"""
        try:
            with open(os.path.join(self.output_dir, self.state_file_name), "r") as f:
                lines = f.readlines()
                return lines[-1].strip() if lines else None
        except Exception as e:
            logger.error(f"Error reading state file: {e}")
            return None

    def get_seed(self) -> str:
        try:
            with open(os.path.join(self.output_dir, self.seed_file_name), "r") as f:
                lines = f.readlines()
                return lines[-1].strip() if lines else None
        except Exception as e:
            logger.error(f"Error reading seed file: {e}")
            return str(self.seed)  # Return the original seed if file read fails

    def get_latest_energy(self) -> float:
        """Get the latest recorded energy value"""
        try:
            with open(os.path.join(self.output_dir, self.energy_file_name), "r") as f:
                lines = f.readlines()[1:]  # Skip header
                if not lines:
                    return None
                last_line = lines[-1]
                return float(last_line.strip().split(',')[1])
        except Exception as e:
            logger.error(f"Error reading energy file: {e}")
            return None

    def configure_commands(
        self, seed: int, system_config: SimulationConfig
    ) -> Dict[str, Any]:
        """Configure simulation objects with optimized settings"""
        state_commands = {}

        for state in self.STATES:
            # Create simulation with optimized settings
            simulation, _ = OpenMMSimulation().create_simulation(
                pdb=self.pdb_obj,
                system_config=system_config.get_config(),
                seed=seed,
            )
            
            # Set CUDA-specific optimizations
            try:
                platform = simulation.context.getPlatform()
                if platform.getName() == "CUDA":
                    # These settings were found to be optimal for most protein simulations
                    simulation.context.setParameter("CudaPrecision", "mixed")  # Balance speed and accuracy
                    
                    # On newer GPUs, these settings can improve performance
                    if hasattr(platform, "setPropertyDefaultValue"):
                        platform.setPropertyDefaultValue("DeviceIndex", "0")  # Use primary GPU
                        platform.setPropertyDefaultValue("Precision", "mixed")
            except Exception as e:
                logger.warning(f"Could not set platform optimizations: {e}")
            
            # Add reporters for state tracking and checkpointing
            simulation.reporters.append(
                LastTwoCheckpointsReporter(
                    file_prefix=f"{self.output_dir}/{state}",
                    reportInterval=self.CHECKPOINT_INTERVAL,
                )
            )
            
            # For monitoring energies - higher reporting frequency for better data
            simulation.reporters.append(
                app.StateDataReporter(
                    file=f"{self.output_dir}/{state}.log",
                    reportInterval=self.STATE_DATA_REPORTER_INTERVAL,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    speed=True,
                )
            )
            
            # Reporter to know when an exit occurs
            simulation.reporters.append(
                ExitFileReporter(
                    filename=f"{self.output_dir}/{state}",
                    reportInterval=self.EXIT_REPORTER_INTERVAL,
                    file_prefix=state,
                )
            )
            
            state_commands[state] = simulation

        return state_commands


class MockSimulationManager(SimulationManager):
    def __init__(self, pdb_id: str, output_dir: str) -> None:
        super().__init__(pdb_id=pdb_id)
        self.required_values = set(["init", "wait", "finished"])
        self.output_dir = output_dir

    def run(self, total_wait_time: int = 1):
        start_time = time.time()

        logger.debug(f"✅ MockSimulationManager.run is running ✅")
        check_if_directory_exists(output_directory=self.output_dir)

        store = os.path.join(self.output_dir, self.state_file_name)
        states = ["init", "wait", "finished"]

        intermediate_interval = total_wait_time / len(states)

        for state in states:
            logger.info(f"Running state: {state}")
            state_time = time.time()
            with open(store, "w") as f:
                f.write(f"{state}\n")

            time.sleep(intermediate_interval)
            logger.info(f"Total state_time: {time.time() - state_time}")

        logger.warning(f"Total run method time: {time.time() - start_time}")
