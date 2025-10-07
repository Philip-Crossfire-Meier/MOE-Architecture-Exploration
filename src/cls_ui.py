# rich imports
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TimeElapsedColumn
from rich_argparse import RichHelpFormatter

from src.app_config import AppConfig

# Configure progress bars
progress_columns = list(Progress.get_default_columns())
_ = progress_columns.pop(3)  # remove the default TextColumn, since we don't need it
progress = Progress(
    *progress_columns,
    TimeElapsedColumn()
)

# Reference: https://github.com/Textualize/rich/discussions/1571
def generate_layout(progress=None, main_text:(str|None)="", loss_error_text:(str|None)="") -> Layout:
    """
    Generate a Rich layout for the UI.
    """
    layout = Layout()
    layout.split(
        Layout(name="status", size=5),
        Layout(ratio=1, name="experts"),
        Layout(size=5, name="loss")
    )
    
    layout["status"].update(Panel(
        progress,
        title="Progress",
        border_style="green",
        padding=(0, 0, 1, 1),
    ))

    layout["experts"].update(Panel(
        main_text,
        title="main",
        border_style="red",
        padding=(1, 1),
    ))

    layout["loss"].update(Panel(
        loss_error_text,
        title="Log",
        border_style="blue",
        padding=(1, 1),
    ))

    return layout

last_expert_utilization = None
last_system_usage = None
last_loss_error_text = None

def update_ui(live:Live, expert_utilization:(str|None)=None, system_usage:(str|None)=None, loss_error_text:(str|None)=None) -> None:
    """
    Update the UI with the latest values.
    """
    # Store the last values
    global last_expert_utilization, last_system_usage, last_loss_error_text
    last_expert_utilization = expert_utilization if expert_utilization is not None else last_expert_utilization
    last_system_usage = system_usage if system_usage is not None else last_system_usage
    last_loss_error_text = loss_error_text if loss_error_text is not None else last_loss_error_text

    main_text = Columns([last_expert_utilization, "Runner Status\n----------------------------\n" + last_system_usage], padding=(1, 1), expand=True)
    live.update(generate_layout(progress, main_text, last_loss_error_text))

    
