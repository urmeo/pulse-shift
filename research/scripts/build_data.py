"""Build the merged hourly analysis panel."""

from pulseshift import panel


def main() -> None:
    hourly_panel = panel.build_panel(write=True)
    active_hours = int(hourly_panel["active_hour"].sum())
    suppressed_hours = int(hourly_panel["suppressed"].sum())
    print(
        f"panel rows: {len(hourly_panel)}  active hours: {active_hours}  suppressed: {suppressed_hours}"
    )
    timestamps = hourly_panel["ts_local"]
    print(f"span: {timestamps.min()} -> {timestamps.max()}")


if __name__ == "__main__":
    main()
