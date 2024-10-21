import logging

import typer

from .process_task_data import process_task_data, process_task_data_callback

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)
app.command()(process_task_data)
app.callback()(process_task_data_callback)

if __name__ == "__main__":
    app()
