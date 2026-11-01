"""Shared Spark session setup for every pipeline job."""
import os
import sys

from pyspark.sql import SparkSession


def get_spark(app_name: str, master: str = "local[6]", driver_memory: str = "4g") -> SparkSession:
    """Create a local Spark session whose workers use this interpreter.

    Without this, Spark starts workers with the `python3` on PATH. On a machine whose
    system Python differs from the project venv, jobs fail with PYTHON_VERSION_MISMATCH.

    The session time zone is UTC on purpose. Divvy timestamps are treated as Chicago
    wall-clock labels, so Spark must never convert them. Under a Chicago session zone,
    labels such as 02:00 on the spring daylight-saving day do not exist and would break.

    Arrow is enabled because the non-Arrow toPandas() path localises timestamps with
    tzlocal(), which crashes on this stack.

    The core and memory limits are deliberate. With every core and a 6 GB driver, the
    features stage exhausted a 14 GB machine and the kernel killed Spark and the editor.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
