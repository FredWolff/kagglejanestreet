"""Utility functions.

This module provides utility functions for executing shell commands
and creating folders.

Functions:
    run_shell_command: Executes a shell command and prints the output in real-time.
    create_folder: Creates a folder, with an option to remove existing ones.
"""

import os
import shutil
import subprocess


def run_shell_command(command: str, cwd: str = None) -> None:
    """Executes a shell command and prints the output/errors in real-time.

    Args:
        command (str): The shell command to execute.
        cwd (str, optional): The working directory where the command should be executed.
                             Defaults to None.
    """
    try:
        # Print the command itself
        print(f"Running command: {command}")

        # Start the process
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )

        # Print the output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Capture any remaining errors
        stderr = process.stderr.read()
        if stderr:
            print("Errors:\n", stderr)

    except Exception as e:
        print(f"An error occurred: {e}")


def create_folder(path: str, rm: bool = False) -> None:
    """Creates a folder.

    Args:
        path (str): Path to the folder.
        rm (bool, optional): Whether to remove the folder if it already exists. Defaults to False.
    """
    if rm:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def rolling_min_periods(window_size: int) -> int:
    """`min_periods` for rolling mean/std over a window that may contain nulls.

    polars' default (`min_periods == window_size`) turns a window into null if it holds even
    one null. Many raw features are structurally null for the first K time_ids of every day
    (K up to 68, see `DataProcessor`), and a 1000-row window spans about a day, so with the
    default their rolling columns come out ~100% null. A lower `min_periods` makes polars
    skip the nulls and compute over the valid values left in the window. Half the window is
    comfortably below the ~930 valid values the worst case leaves, and above the 2 needed
    for a std.

    Args:
        window_size (int): Rolling window size, in rows.

    Returns:
        int: Minimum number of non-null values a window needs to produce a value.
    """
    return max(window_size // 2, 2)
