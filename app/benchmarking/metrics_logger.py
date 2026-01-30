# app/benchmarking/metrics_logger.py
"""
Enhanced metrics logging system for Keeling Schedule generation that uses a SQLite database
to store structured benchmarking data about schedules, amendments, and prompts.
"""
import os
import sqlite3
import re
import math
from datetime import datetime
from typing import Dict, Any, Optional, Union
from semantic_kernel.functions import FunctionResult, KernelFunctionFromPrompt
from ..logging.debug_logger import get_logger

logger = get_logger(__name__)


class MetricsLogger:
    """
    Structured metrics logger for Keeling Schedule operations.

    Tracks the complete lifecycle of schedule generation including:
    - Schedule metadata and timing
    - Amendment identification and application
    - LLM prompt execution and costs
    """

    def __init__(
        self,
        enable_aws_bedrock: bool = False,
        bedrock_service_id: str = None,
        bedrock_model_id: str = None,
        enable_azure_openai: bool = False,
        azure_model_deployment_name: str = None,
        db_path: str = "benchmarking_data/keeling_metrics.db",
    ):
        """
        Initialise the logger with service configuration and database connection.

        Args:
            enable_aws_bedrock: Whether AWS Bedrock service is enabled
            bedrock_service_id: AWS Bedrock service identifier
            bedrock_model_id: AWS Bedrock model identifier
            enable_azure_openai: Whether Azure OpenAI service is enabled
            azure_model_deployment_name: Azure OpenAI deployment name
            db_path: Path to SQLite database file
        """
        self.enable_aws_bedrock = enable_aws_bedrock
        self.bedrock_service_id = bedrock_service_id
        self.bedrock_model_id = bedrock_model_id
        self.enable_azure_openai = enable_azure_openai
        self.azure_model_deployment_name = azure_model_deployment_name

        # Database configuration
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialise database with schema
        self._initialise_database()

        # Auto-load ground truth data if available
        self._auto_load_ground_truth()

        # For tracking ongoing operations
        self._current_operations = {}

    # ==================== Public Interface Methods ====================

    def log_schedule_start(
        self,
        schedule_id: str,
        act_name: str,
        model_id: str,
        service_id: str,
        max_worker_threads: int,
        bill_xml_size: Optional[int] = None,
        act_xml_size: Optional[int] = None,
    ) -> None:
        """
        Log the start of a Keeling schedule generation process.

        Args:
            schedule_id: Unique identifier for this schedule
            act_name: Name of the act being amended
            model_id: Identifier for the LLM model used
            service_id: The service providing the LLM capability
            max_worker_threads: Maximum number of worker threads for parallel processing
            bill_xml_size: Size of input bill XML in bytes
            act_xml_size: Size of input act XML in bytes
        """
        start_ts = datetime.utcnow().isoformat()

        # Store in memory for duration tracking
        self._current_operations[schedule_id] = {"start_timestamp": start_ts, "amendments": {}}

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO schedules (
                    schedule_id, act_name, model_id, service_id, max_worker_threads,
                    start_timestamp, bill_xml_size, act_xml_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    act_name,
                    model_id,
                    service_id,
                    max_worker_threads,
                    start_ts,
                    bill_xml_size,
                    act_xml_size,
                ),
            )

            conn.commit()
            conn.close()
            logger.info(f"Started logging for schedule {schedule_id}")
        except Exception as e:
            logger.error(f"Failed to log schedule start: {e}")

    def log_schedule_end(self, schedule_id: str, total_amendments_found: int, total_amendments_applied: int) -> None:
        """
        Log the completion of a Keeling schedule generation process.

        Args:
            schedule_id: Unique identifier for this schedule
            total_amendments_found: Total number of amendments identified
            total_amendments_applied: Total number of amendments applied
        """
        end_ts = datetime.utcnow().isoformat()

        # Calculate metrics
        duration_seconds = self._calculate_duration(schedule_id, end_ts)
        application_rate = self._calculate_application_rate(total_amendments_found, total_amendments_applied)

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Update schedule with completion info
            cursor.execute(
                """
                UPDATE schedules SET
                    end_timestamp = ?,
                    total_duration_seconds = ?,
                    total_amendments_found = ?,
                    total_amendments_applied = ?,
                    application_rate = ?
                WHERE schedule_id = ?
                """,
                (
                    end_ts,
                    duration_seconds,
                    total_amendments_found,
                    total_amendments_applied,
                    application_rate,
                    schedule_id,
                ),
            )

            # Update aggregated metrics
            self._update_schedule_metrics(cursor, schedule_id)

            conn.commit()
            conn.close()

            # Clear from memory
            if schedule_id in self._current_operations:
                del self._current_operations[schedule_id]

            logger.info(f"Completed logging for schedule {schedule_id}")
        except Exception as e:
            logger.error(f"Failed to log schedule end: {e}")

    def log_amendment(
        self,
        schedule_id: str,
        amendment_id: str,
        source: str,
        source_eid: str,
        affected_provision: str,
        location: str,
        amendment_type: str,
        whole_provision: bool,
        identification_time_seconds: Optional[float] = None,
    ) -> None:
        """
        Log information about an identified amendment.

        Args:
            schedule_id: Parent schedule identifier
            amendment_id: Unique identifier for this amendment
            source: Reference to the provision containing the amendment
            source_eid: XML element ID of the source provision
            affected_provision: The provision being modified
            location: Where to apply the amendment (BEFORE, AFTER, REPLACE)
            amendment_type: Type of modification (INSERTION, DELETION, SUBSTITUTION)
            whole_provision: Boolean flag indicating if it affects an entire section
            identification_time_seconds: Time taken to identify the amendment
        """
        # Store amendment start time for later application timing
        if schedule_id in self._current_operations:
            self._current_operations[schedule_id]["amendments"][amendment_id] = {"start_time": datetime.utcnow()}

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO amendments (
                    amendment_id, schedule_id, source, source_eid, affected_provision,
                    location, amendment_type, whole_provision, identification_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    amendment_id,
                    schedule_id,
                    source,
                    source_eid,
                    affected_provision,
                    location,
                    amendment_type,
                    1 if whole_provision else 0,
                    identification_time_seconds,
                ),
            )

            conn.commit()
            conn.close()
            logger.debug(f"Logged amendment {amendment_id} for schedule {schedule_id}")
        except Exception as e:
            logger.error(f"Failed to log amendment: {e}")

    def update_amendment_application(
        self, amendment_id: str, application_time_seconds: Optional[float] = None, application_status: bool = False
    ) -> None:
        """
        Update amendment information after application attempt.

        Args:
            amendment_id: Unique identifier for this amendment
            application_time_seconds: Time taken to apply the amendment
            application_status: Whether the amendment was applied
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Get identification time to calculate total time
            cursor.execute(
                "SELECT identification_time_seconds, schedule_id FROM amendments WHERE amendment_id = ?",
                (amendment_id,),
            )

            result = cursor.fetchone()
            if not result:
                logger.error(f"Amendment {amendment_id} not found for update")
                conn.close()
                return

            identification_time, schedule_id = result

            # Calculate total processing time
            total_time = None
            if identification_time is not None and application_time_seconds is not None:
                total_time = identification_time + application_time_seconds

            # Update amendment record
            cursor.execute(
                """
                UPDATE amendments SET
                    application_time_seconds = ?,
                    total_processing_time_seconds = ?,
                    application_status = ?
                WHERE amendment_id = ?
                """,
                (application_time_seconds, total_time, 1 if application_status else 0, amendment_id),
            )

            conn.commit()
            conn.close()

            # Clear from memory if needed
            self._clear_amendment_from_memory(schedule_id, amendment_id)

            logger.debug(f"Updated application status for amendment {amendment_id}")
        except Exception as e:
            logger.error(f"Failed to update amendment application: {e}")

    def update_schedule_act_size(self, schedule_id: str, act_xml_size: int) -> None:
        """
        Update the act XML size for a schedule.

        Args:
            schedule_id: Schedule identifier
            act_xml_size: Size of the act XML in bytes
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE schedules SET act_xml_size = ? WHERE schedule_id = ?", (act_xml_size, schedule_id))
            conn.commit()
            conn.close()
            logger.debug(f"Updated act_xml_size for schedule {schedule_id}")
        except Exception as e:
            logger.error(f"Failed to update act_xml_size: {e}")

    def log_prompt(
        self,
        prompt_id: str,
        prompt: KernelFunctionFromPrompt,
        prompt_output: str,
        input_parameters: dict,
        schedule_id: str,
        prompt_start_ts: str,
        prompt_end_ts: str,
        amendment_id: Optional[str] = None,
        prompt_category: Optional[str] = None,
    ) -> None:
        """
        Log information about a prompt execution.

        Args:
            prompt_id: Unique identifier for this prompt execution
            prompt: The kernel function that was executed
            prompt_output: The output from the prompt (unused but kept for interface)
            input_parameters: Parameters passed to the prompt
            schedule_id: Schedule this prompt belongs to
            prompt_start_ts: Start timestamp
            prompt_end_ts: End timestamp
            amendment_id: Optional amendment this prompt relates to
            prompt_category: Category of prompt (identification/application)
        """
        # Extract metadata and calculate timing
        metadata = self._extract_metadata(prompt, prompt_output, input_parameters)
        inference_time = self._calculate_inference_time(prompt_start_ts, prompt_end_ts)

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Prepare token values for database
            prompt_tokens = self._prepare_token_value(metadata["prompt_tokens"])
            completion_tokens = self._prepare_token_value(metadata["completion_tokens"])
            total_tokens = self._prepare_token_value(metadata["total_tokens"])

            cursor.execute(
                """
                INSERT INTO prompts (
                    prompt_id, schedule_id, amendment_id, prompt_name, prompt_template,
                    model_id, prompt_category, start_timestamp, end_timestamp,
                    prompt_tokens, completion_tokens, total_tokens,
                    inference_time_seconds, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt_id,
                    schedule_id,
                    amendment_id,
                    prompt.name,
                    prompt.name,  # Using name as template for now
                    metadata["model_id"],
                    prompt_category,
                    prompt_start_ts,
                    prompt_end_ts,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    inference_time,
                    metadata["cost_usd"],
                ),
            )

            conn.commit()
            conn.close()
            logger.debug(f"Successfully logged prompt {prompt_id} for schedule {schedule_id}")
        except Exception as e:
            logger.error(f"Failed to log prompt: {str(e)}")
            logger.exception("Full exception details:")

    # ==================== Private Database Methods ====================

    def _initialise_database(self) -> None:
        """Initialise SQLite database with the schema for the three main entities."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Create tables
            self._create_schedules_table(cursor)
            self._create_amendments_table(cursor)
            self._create_prompts_table(cursor)
            self._create_ground_truth_table(cursor)

            conn.commit()
            conn.close()
            logger.info("Database initialised successfully")
        except Exception as e:
            logger.error(f"Failed to initialise database: {e}")

    def _create_schedules_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the schedules table."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schedules (
                schedule_id TEXT PRIMARY KEY,
                act_name TEXT,
                model_id TEXT,
                service_id TEXT,
                max_worker_threads INTEGER,
                start_timestamp TEXT,
                end_timestamp TEXT,
                total_duration_seconds REAL,
                bill_xml_size INTEGER,
                act_xml_size INTEGER,
                total_amendments_found INTEGER,
                total_amendments_applied INTEGER,
                application_rate REAL,
                total_prompts_executed INTEGER,
                total_token_usage INTEGER,
                total_cost_usd REAL,
                dataset_name TEXT,
                identification_precision REAL,
                identification_recall REAL,
                identification_f1 REAL,
                location_accuracy REAL,
                whole_provision_accuracy REAL,
                insertion_application_rate REAL,
                deletion_application_rate REAL,
                substitution_application_rate REAL,
                geometric_mean_application REAL
            )
        """
        )

    def _create_amendments_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the amendments table."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS amendments (
                amendment_id TEXT PRIMARY KEY,
                schedule_id TEXT,
                source TEXT,
                source_eid TEXT,
                affected_provision TEXT,
                location TEXT,
                amendment_type TEXT,
                whole_provision INTEGER,
                identification_time_seconds REAL,
                application_time_seconds REAL,
                total_processing_time_seconds REAL,
                application_status INTEGER,
                FOREIGN KEY (schedule_id) REFERENCES schedules (schedule_id)
            )
        """
        )

    def _create_prompts_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the prompts table."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                schedule_id TEXT,
                amendment_id TEXT,
                prompt_name TEXT,
                prompt_template TEXT,
                model_id TEXT,
                prompt_category TEXT,
                start_timestamp TEXT,
                end_timestamp TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                inference_time_seconds REAL,
                cost_usd REAL,
                FOREIGN KEY (schedule_id) REFERENCES schedules (schedule_id),
                FOREIGN KEY (amendment_id) REFERENCES amendments (amendment_id)
            )
        """
        )

    def _create_ground_truth_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the ground_truth table."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ground_truth (
                ground_truth_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_eid TEXT NOT NULL,
                type_of_amendment TEXT NOT NULL,
                affected_provision TEXT NOT NULL,
                location TEXT NOT NULL,
                whole_provision INTEGER NOT NULL
            )
        """
        )

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    # ==================== Private Calculation Methods ====================

    def _calculate_duration(self, schedule_id: str, end_ts: str) -> Optional[float]:
        """Calculate duration in seconds from start to end timestamp."""
        if schedule_id not in self._current_operations:
            return None

        start_ts = self._current_operations[schedule_id].get("start_timestamp")
        if not start_ts:
            return None

        try:
            start_dt = datetime.fromisoformat(start_ts)
            end_dt = datetime.fromisoformat(end_ts)
            return (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            return None

    def _calculate_application_rate(self, total_found: int, total_applied: int) -> Optional[float]:
        """
        Calculate application rate as a percentage.

        Args:
            total_found: Total amendments identified
            total_applied: Total amendments applied

        Returns:
            Application rate as percentage, or None if no amendments found
        """
        if total_found > 0:
            return (total_applied / total_found) * 100
        return None

    def _calculate_inference_time(self, start_ts: str, end_ts: str) -> Optional[float]:
        """Calculate inference time from timestamps."""
        if not start_ts or not end_ts:
            return None

        try:
            start = datetime.fromisoformat(start_ts)
            end = datetime.fromisoformat(end_ts)
            return (end - start).total_seconds()
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating inference time: {e}")
            return None

    def _calculate_cost(
        self, prompt_tokens: Union[int, float], completion_tokens: Union[int, float], service_id: Optional[str] = None
    ) -> float:
        """
        Calculate the estimated cost based on token usage.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            service_id: The service ID to determine which pricing to use

        Returns:
            Estimated cost in USD
        """
        # Validate inputs
        if not isinstance(prompt_tokens, (int, float)) or not isinstance(completion_tokens, (int, float)):
            return 0.0

        # Set default rates for Azure OpenAI GPT-4o
        input_rate = 0.005  # $0.0050 per 1K input tokens
        output_rate = 0.015  # $0.0150 per 1K output tokens

        # Use different rates for Bedrock Claude
        if service_id == self.bedrock_service_id:
            input_rate = 3.00 / 1000  # $3.00 per 1M input tokens (= $0.003 per 1K)
            output_rate = 15.00 / 1000  # $15.00 per 1M output tokens (= $0.015 per 1K)

        # Calculate cost
        input_cost = (prompt_tokens / 1000) * input_rate
        output_cost = (completion_tokens / 1000) * output_rate
        total_cost = input_cost + output_cost

        return round(total_cost, 6)  # Round to 6 decimal places for currency

    # ==================== Private Helper Methods ====================

    def _update_schedule_metrics(self, cursor: sqlite3.Cursor, schedule_id: str) -> None:
        """Update aggregated metrics for a schedule."""
        cursor.execute(
            """
            SELECT
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost,
                COUNT(*) as total_prompts
            FROM prompts
            WHERE schedule_id = ?
            """,
            (schedule_id,),
        )

        result = cursor.fetchone()
        if result:
            total_tokens, total_cost, total_prompts = result

            cursor.execute(
                """
                UPDATE schedules SET
                    total_token_usage = ?,
                    total_cost_usd = ?,
                    total_prompts_executed = ?
                WHERE schedule_id = ?
                """,
                (total_tokens, total_cost, total_prompts, schedule_id),
            )

    def _extract_metadata(
        self, prompt: KernelFunctionFromPrompt, prompt_output: FunctionResult, input_parameters: dict
    ) -> Dict[str, Any]:
        """Extract metadata from prompt execution results and input parameters."""
        # Determine which service was used
        service_id = self._determine_service_id(prompt)

        # Initialise defaults
        metadata = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "model_id": "N/A",
            "cost_usd": 0.0,
        }

        # Extract token information from input parameters
        if "extracted_prompt_tokens" in input_parameters and input_parameters["extracted_prompt_tokens"] is not None:
            metadata["prompt_tokens"] = input_parameters["extracted_prompt_tokens"]

        if (
            "extracted_completion_tokens" in input_parameters
            and input_parameters["extracted_completion_tokens"] is not None
        ):
            metadata["completion_tokens"] = input_parameters["extracted_completion_tokens"]

        if "extracted_total_tokens" in input_parameters and input_parameters["extracted_total_tokens"] is not None:
            metadata["total_tokens"] = input_parameters["extracted_total_tokens"]

        # Extract model ID
        if "extracted_model_id" in input_parameters and input_parameters["extracted_model_id"] is not None:
            metadata["model_id"] = input_parameters["extracted_model_id"]
        elif service_id == self.bedrock_service_id:
            metadata["model_id"] = self.bedrock_model_id
        elif service_id == self.azure_model_deployment_name:
            metadata["model_id"] = self.azure_model_deployment_name

        # Calculate cost if we have token information
        if isinstance(metadata["prompt_tokens"], (int, float)) and isinstance(
            metadata["completion_tokens"], (int, float)
        ):
            metadata["cost_usd"] = self._calculate_cost(
                metadata["prompt_tokens"], metadata["completion_tokens"], service_id
            )

        return metadata

    def _determine_service_id(self, prompt: KernelFunctionFromPrompt) -> Optional[str]:
        """Determine which service was used for this prompt."""
        # Simple case: only one service enabled
        if self.enable_aws_bedrock and not self.enable_azure_openai:
            return self.bedrock_service_id
        elif self.enable_azure_openai and not self.enable_aws_bedrock:
            return self.azure_model_deployment_name

        # Both services enabled or both disabled, need to check settings
        if hasattr(prompt, "prompt_execution_settings"):
            # Check for Azure OpenAI settings first
            azure_settings = prompt.prompt_execution_settings.get(self.azure_model_deployment_name, None)
            if azure_settings:
                return self.azure_model_deployment_name

            # Check for Bedrock settings
            bedrock_settings = prompt.prompt_execution_settings.get(
                self.bedrock_service_id, None
            ) or prompt.prompt_execution_settings.get("bedrock-claude", None)
            if bedrock_settings:
                return self.bedrock_service_id

        return None

    def _prepare_token_value(self, value: Any) -> Optional[int]:
        """Convert token value to integer or None for database storage."""
        if value is not None:
            return int(value)
        return None

    def _clear_amendment_from_memory(self, schedule_id: str, amendment_id: str) -> None:
        """Clear amendment from in-memory tracking."""
        if (
            schedule_id in self._current_operations
            and amendment_id in self._current_operations[schedule_id]["amendments"]
        ):
            del self._current_operations[schedule_id]["amendments"][amendment_id]

    def load_ground_truth_if_needed(self, dataset_name: str, csv_path: str) -> None:
        """Load pre-cleaned ground truth CSV into database if not already loaded."""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        # Check if already loaded
        cursor.execute("SELECT COUNT(*) FROM ground_truth WHERE dataset_name = ?", (dataset_name,))
        if cursor.fetchone()[0] > 0:
            logger.debug(f"Ground truth '{dataset_name}' already loaded")
            conn.close()
            return

        # Load CSV
        import csv

        # Generate unique IDs
        import uuid

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            loaded_count = 0

            for row in reader:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO ground_truth
                    (ground_truth_id, dataset_name, source, source_eid, type_of_amendment,
                    affected_provision, location, whole_provision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        dataset_name,
                        row["source"],
                        row["source_eid"],
                        row["type_of_amendment"],
                        row["affected_provision"],
                        row["location"],
                        1 if row["whole_provision"].strip().upper() == "TRUE" else 0,
                    ),
                )
                loaded_count += 1

        conn.commit()
        logger.info(f"Loaded {loaded_count} ground truth amendments for '{dataset_name}'")
        conn.close()

    def _auto_load_ground_truth(self) -> None:
        """Auto-load any CSV files in the ground truth directory."""
        ground_truth_dir = os.path.join(os.path.dirname(self.db_path), "ground_truth")

        if not os.path.exists(ground_truth_dir):
            logger.debug(f"Ground truth directory does not exist: {ground_truth_dir}")
            return

        for filename in os.listdir(ground_truth_dir):
            if filename.endswith(".csv"):
                # Remove .csv extension
                dataset_name = filename[:-4]

                csv_path = os.path.join(ground_truth_dir, filename)

                try:
                    self.load_ground_truth_if_needed(dataset_name, csv_path)
                except Exception as e:
                    logger.error(f"Failed to load ground truth {filename}: {e}")

    def evaluate_schedule_accuracy(self, schedule_id: str, act_name: str) -> Dict[str, Any]:
        """
        Evaluate schedule accuracy against ground truth if available.

        Args:
            schedule_id: Schedule to evaluate
            act_name: Name of the act being amended

        Returns:
            Dictionary of evaluation metrics (empty if no ground truth available)
        """
        # Try to match dataset based on act name
        dataset_name = self._get_dataset_name_from_act(act_name)

        if not dataset_name:
            logger.debug(f"No ground truth dataset matched for act '{act_name}'")
            return {}

        conn = self._get_db_connection()
        cursor = conn.cursor()

        # Check if ground truth exists
        cursor.execute("SELECT COUNT(*) FROM ground_truth WHERE dataset_name = ?", (dataset_name,))
        if cursor.fetchone()[0] == 0:
            logger.debug(f"No ground truth available for dataset '{dataset_name}'")
            conn.close()
            return {}

        # Calculate identification metrics
        ident_metrics = self._calculate_identification_metrics(cursor, schedule_id, dataset_name)

        # Calculate metadata accuracy metrics (location, whole_provision)
        metadata_metrics = self._calculate_metadata_accuracy(cursor, schedule_id, dataset_name)

        # Calculate application rates (only for correctly identified amendments)
        application_metrics = self._calculate_application_rates_by_type(cursor, schedule_id, dataset_name)

        # Combine all metrics
        all_metrics = {**ident_metrics, **metadata_metrics, **application_metrics}

        # Update schedule record with metrics
        self._update_schedule_evaluation_metrics(cursor, schedule_id, dataset_name, all_metrics)

        conn.commit()

        # Log results
        self._log_evaluation_results(schedule_id, all_metrics)

        conn.close()

        return all_metrics

    def _calculate_identification_metrics(
        self, cursor: sqlite3.Cursor, schedule_id: str, dataset_name: str
    ) -> Dict[str, Any]:
        """
        Calculate precision, recall, and F1 for amendment identification.

        Also includes lists of false positives and false negatives for debugging.

        Args:
            cursor: Database cursor
            schedule_id: Schedule to evaluate
            dataset_name: Name of the ground truth dataset

        Returns:
            Dictionary containing precision, recall, F1, raw counts, and FP/FN details
        """
        # True positives: correctly identified amendments
        cursor.execute(
            """
            SELECT COUNT(DISTINCT a.amendment_id)
            FROM amendments a
            JOIN ground_truth gt ON
                a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            WHERE a.schedule_id = ? AND gt.dataset_name = ?
        """,
            (schedule_id, dataset_name),
        )
        true_positives = cursor.fetchone()[0]

        # False positives: system amendments that don't match any ground truth in this dataset
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM amendments a
            WHERE a.schedule_id = ?
            AND NOT EXISTS (
                SELECT 1 FROM ground_truth gt
                WHERE gt.dataset_name = ?
                AND a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            )
        """,
            (schedule_id, dataset_name),
        )
        false_positives = cursor.fetchone()[0]

        # False negatives: ground truth not found by system
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM ground_truth gt
            WHERE gt.dataset_name = ?
            AND NOT EXISTS (
                SELECT 1 FROM amendments a
                WHERE a.schedule_id = ?
                AND a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            )
        """,
            (dataset_name, schedule_id),
        )
        false_negatives = cursor.fetchone()[0]

        # Get the false positive details
        cursor.execute(
            """
            SELECT source_eid, affected_provision, amendment_type
            FROM amendments a
            WHERE a.schedule_id = ?
            AND NOT EXISTS (
                SELECT 1 FROM ground_truth gt
                WHERE gt.dataset_name = ?
                AND a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            )
            ORDER BY source_eid
        """,
            (schedule_id, dataset_name),
        )

        false_positive_details = [f"{row[0]} -> {row[1]} ({row[2]})" for row in cursor.fetchall()]

        # Get the false negative details
        cursor.execute(
            """
            SELECT source_eid, affected_provision, type_of_amendment
            FROM ground_truth gt
            WHERE gt.dataset_name = ?
            AND NOT EXISTS (
                SELECT 1 FROM amendments a
                WHERE a.schedule_id = ?
                AND a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            )
            ORDER BY source_eid
        """,
            (dataset_name, schedule_id),
        )

        false_negative_details = [f"{row[0]} -> {row[1]} ({row[2]})" for row in cursor.fetchall()]

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "false_positive_details": false_positive_details,
            "false_negative_details": false_negative_details,
        }

    def _calculate_metadata_accuracy(
        self, cursor: sqlite3.Cursor, schedule_id: str, dataset_name: str
    ) -> Dict[str, float]:
        """
        Calculate accuracy of location and whole_provision metadata for correctly identified amendments.

        Args:
            cursor: Database cursor
            schedule_id: Schedule to evaluate
            dataset_name: Name of the ground truth dataset

        Returns:
            Dictionary containing location_accuracy and whole_provision_accuracy
        """
        # Location accuracy
        cursor.execute(
            """
            SELECT
                COUNT(CASE WHEN LOWER(a.location) = LOWER(gt.location) THEN 1 END) as correct_location,
                COUNT(*) as total_matched
            FROM amendments a
            JOIN ground_truth gt ON
                a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            WHERE a.schedule_id = ? AND gt.dataset_name = ?
        """,
            (schedule_id, dataset_name),
        )
        row = cursor.fetchone()
        location_accuracy = row[0] / row[1] if row[1] > 0 else 0

        # Whole provision accuracy
        cursor.execute(
            """
            SELECT
                COUNT(CASE WHEN a.whole_provision = gt.whole_provision THEN 1 END) as correct_whole,
                COUNT(*) as total_matched
            FROM amendments a
            JOIN ground_truth gt ON
                a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            WHERE a.schedule_id = ? AND gt.dataset_name = ?
        """,
            (schedule_id, dataset_name),
        )
        row = cursor.fetchone()
        whole_provision_accuracy = row[0] / row[1] if row[1] > 0 else 0

        return {"location_accuracy": location_accuracy, "whole_provision_accuracy": whole_provision_accuracy}

    def _calculate_application_rates_by_type(
        self, cursor: sqlite3.Cursor, schedule_id: str, dataset_name: str
    ) -> Dict[str, float]:
        """
        Calculate application rates by amendment type for correctly identified amendments only.

        Note: This measures whether the system successfully applied amendments without errors,
        not whether those applications are semantically correct (which would require manual
        verification of the actual text transformations).

        Args:
            cursor: Database cursor
            schedule_id: Schedule to evaluate
            dataset_name: Name of the ground truth dataset

        Returns:
            Dictionary containing application rates by type and geometric mean
        """
        # Get application rates only for amendments that match ground truth
        cursor.execute(
            """
            SELECT
                LOWER(a.amendment_type) as type_normalized,
                SUM(CASE WHEN a.application_status = 1 THEN 1 ELSE 0 END) as successful,
                COUNT(*) as total
            FROM amendments a
            JOIN ground_truth gt ON
                a.source_eid = gt.source_eid
                AND a.affected_provision = gt.affected_provision
                AND LOWER(a.amendment_type) = LOWER(gt.type_of_amendment)
            WHERE a.schedule_id = ? AND gt.dataset_name = ?
            GROUP BY LOWER(a.amendment_type)
        """,
            (schedule_id, dataset_name),
        )

        application_rates = {}
        for row in cursor.fetchall():
            type_name = row[0] if row[0] else "unknown"
            rate = row[1] / row[2] if row[2] > 0 else 0
            application_rates[f"{type_name}_application_rate"] = rate

        # Calculate geometric mean of application rates
        rates_list = [v for v in application_rates.values() if v > 0]
        geometric_mean = math.pow(math.prod(rates_list), 1 / len(rates_list)) if rates_list else 0

        return {**application_rates, "geometric_mean_application": geometric_mean}

    def _update_schedule_evaluation_metrics(
        self, cursor: sqlite3.Cursor, schedule_id: str, dataset_name: str, metrics: Dict[str, float]
    ) -> None:
        """
        Update the schedules table with evaluation metrics.

        Args:
            cursor: Database cursor
            schedule_id: Schedule to update
            dataset_name: Name of the ground truth dataset
            metrics: Dictionary of calculated metrics

        Returns:
            None
        """
        cursor.execute(
            """
            UPDATE schedules SET
                dataset_name = ?,
                identification_precision = ?,
                identification_recall = ?,
                identification_f1 = ?,
                location_accuracy = ?,
                whole_provision_accuracy = ?,
                insertion_application_rate = ?,
                deletion_application_rate = ?,
                substitution_application_rate = ?,
                geometric_mean_application = ?
            WHERE schedule_id = ?
        """,
            (
                dataset_name,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1", 0),
                metrics.get("location_accuracy", 0),
                metrics.get("whole_provision_accuracy", 0),
                metrics.get("insertion_application_rate", 0),
                metrics.get("deletion_application_rate", 0),
                metrics.get("substitution_application_rate", 0),
                metrics.get("geometric_mean_application", 0),
                schedule_id,
            ),
        )

    def _log_evaluation_results(self, schedule_id: str, metrics: Dict[str, float]) -> None:
        """
        Log evaluation results to the logger.

        Args:
            schedule_id: Schedule that was evaluated
            metrics: Dictionary of calculated metrics

        Returns:
            None
        """
        # Log the summary
        logger.info(
            f"Evaluation for schedule {schedule_id}: "
            f"Precision={metrics.get('precision', 0):.3f}, "
            f"Recall={metrics.get('recall', 0):.3f}, "
            f"F1={metrics.get('f1', 0):.3f}, "
            f"Location Acc={metrics.get('location_accuracy', 0):.3f}, "
            f"Whole Provision Acc={metrics.get('whole_provision_accuracy', 0):.3f}, "
            f"Geometric Mean App={metrics.get('geometric_mean_application', 0):.3f} "
            f"(TP={metrics.get('true_positives', 0)}, "
            f"FP={metrics.get('false_positives', 0)}, "
            f"FN={metrics.get('false_negatives', 0)})"
        )

        # Log false positive details if present
        fp_details = metrics.get("false_positive_details", [])
        if fp_details:
            logger.info(f"  False positives ({len(fp_details)}):")
            for detail in fp_details:
                logger.info(f"    - {detail}")

        # Log false negative details if present
        fn_details = metrics.get("false_negative_details", [])
        if fn_details:
            logger.info(f"  False negatives ({len(fn_details)}):")
            for detail in fn_details:
                logger.info(f"    - {detail}")

    def _get_dataset_name_from_act(self, act_name: str) -> Optional[str]:
        """
        Match act name to available ground truth datasets using normalisation.

        Args:
            act_name: Name of the act being amended

        Returns:
            Dataset name if matched, None otherwise
        """

        def normalise(text: str) -> str:
            """Normalise text for matching."""
            # Convert to lowercase, remove special chars, compress spaces
            normalised = text.lower()
            normalised = re.sub(r"[^\w\s]", "", normalised)
            normalised = re.sub(r"\s+", "_", normalised)
            return normalised.strip("_")

        # Normalise the act name
        normalised_act = normalise(act_name)

        # Remove "act" for better matching
        normalised_act = re.sub(r"_act_|_act$|^act_", "_", normalised_act)
        normalised_act = re.sub(r"_+", "_", normalised_act).strip("_")

        # Get available ground truth datasets
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT dataset_name FROM ground_truth")
        datasets = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Check if normalised act name is contained in any dataset name
        for dataset in datasets:
            if normalised_act in dataset:
                logger.debug(f"Matched act '{act_name}' (normalised: '{normalised_act}') to dataset '{dataset}'")
                return dataset

        return None
