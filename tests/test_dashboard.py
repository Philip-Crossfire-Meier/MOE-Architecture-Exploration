from src.cls_ui import update_ui, generate_layout

def test_generate_layout() -> None:
    # Test layout generation
    layout = generate_layout(
        progress=None,
        main_text="Main Text",
        loss_error_text="Loss/Error Text"
    )
    assert layout is not None

def test_update_ui() -> None:
    # Test UI update function
    from rich.live import Live
    from rich.text import Text
    
    live = Live(Text("Initial"), refresh_per_second=4)
    live.start()
    try:
        update_ui(
            live,
            expert_utilization="Expert Utilization",
            system_usage="System Usage",
            loss_error_text="Loss/Error Text"
        )
    finally:
        live.stop()

 