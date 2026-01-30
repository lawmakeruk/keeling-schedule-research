# app/benchmarking/metrics_logger.py
"""
Enhanced metrics logging system for Keeling Schedule generation that uses a SQLite database
to store structured benchmarking data about schedules, amendments, and prompts.
"""

import os
import sqlite3
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

        # For tracking ongoing operations
        self._current_operations = {}

    # ==================== Public Interface Methods ====================

    def log_schedule_start(
        self,
        schedule_id: str,
        act_name: str,
        model_id: str,
        service_id: str,
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
                    schedule_id, act_name, model_id, service_id, start_timestamp,
                    bill_xml_size, act_xml_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (schedule_id, act_name, model_id, service_id, start_ts, bill_xml_size, act_xml_size),
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
            total_amendments_applied: Total number of amendments successfully applied
        """
        end_ts = datetime.utcnow().isoformat()

        # Calculate metrics
        duration_seconds = self._calculate_duration(schedule_id, end_ts)
        amendment_success_rate = self._calculate_success_rate(total_amendments_found, total_amendments_applied)

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
                    amendment_success_rate = ?
                WHERE schedule_id = ?
                """,
                (
                    end_ts,
                    duration_seconds,
                    total_amendments_found,
                    total_amendments_applied,
                    amendment_success_rate,
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
        self, amendment_id: str, application_time_seconds: Optional[float] = None, success_status: bool = False
    ) -> None:
        """
        Update amendment information after application attempt.

        Args:
            amendment_id: Unique identifier for this amendment
            application_time_seconds: Time taken to apply the amendment
            success_status: Whether the amendment was successfully applied
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
                    success_status = ?
                WHERE amendment_id = ?
                """,
                (application_time_seconds, total_time, 1 if success_status else 0, amendment_id),
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

            conn.commit()
            conn.close()
            logger.info("Database initialised successfully")
        except Exception as e:
            logger.error(f"Failed to initialise database: {e}")

    def _create_schedules_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the schedules table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                schedule_id TEXT PRIMARY KEY,
                act_name TEXT,
                model_id TEXT,
                service_id TEXT,
                start_timestamp TEXT,
                end_timestamp TEXT,
                total_duration_seconds REAL,
                bill_xml_size INTEGER,
                act_xml_size INTEGER,
                total_amendments_found INTEGER,
                total_amendments_applied INTEGER,
                amendment_success_rate REAL,
                total_prompts_executed INTEGER,
                total_token_usage INTEGER,
                total_cost_usd REAL
            )
        """)

    def _create_amendments_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the amendments table."""
        cursor.execute("""
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
                success_status INTEGER,
                FOREIGN KEY (schedule_id) REFERENCES schedules (schedule_id)
            )
        """)

    def _create_prompts_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the prompts table."""
        cursor.execute("""
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
        """)

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

    def _calculate_success_rate(self, total_found: int, total_applied: int) -> Optional[float]:
        """Calculate amendment success rate as a percentage."""
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
