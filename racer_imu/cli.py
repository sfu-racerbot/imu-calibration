"""Shared command-line plumbing for tools/*.py."""
from __future__ import annotations

import argparse
import atexit
from pathlib import Path

from .frames import Config, load_config
from .pipeline import make_sources
from .sim import SCENARIOS, SimServer


def add_common_args(ap: argparse.ArgumentParser, sim_default: str | None = None):
    ap.add_argument("--config", default=None, help="car.yaml (default: config/car.yaml)")
    ap.add_argument("--vesc", default=None, help="VESC serial port (overrides car.yaml)")
    ap.add_argument("--bno", default=None, help="BNO085 MCU serial port (overrides car.yaml)")
    ap.add_argument("--sim", nargs="?", const=sim_default or "drift", default=None, choices=SCENARIOS,
                    help="use the built-in simulator instead of hardware (optionally pick a scenario)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="IMU names to use (default: all in car.yaml, or just the one whose port you passed)")
    return ap


def pick_only(args, only):
    """Explicit `only` wins; else --only; else if exactly one of --vesc/--bno was given, use just that IMU."""
    if only:
        return only
    if getattr(args, "only", None):
        return args.only
    given = [n for n in ("vesc", "bno") if getattr(args, n, None)]
    return given if len(given) == 1 else None


def use_sim_paths(cfg: Config):
    """In sim mode, read/write calibration under calib/sim/ so real calibration files are never touched."""
    for ex in cfg.imus.values():
        if ex.extra.get("calib"):
            ex.extra["calib"] = str(Path("calib/sim") / Path(ex.extra["calib"]).name)
    sc = cfg.raw.setdefault("sync", {})
    sc["file"] = str(Path("calib/sim") / Path(sc.get("file", "calib/sync.json")).name)


def setup(args, only: list[str] | None = None, scenario: str | None = None, processes: bool = False):
    """Returns (cfg, sources, sim_or_None). Sources are not started yet.

    processes=True runs the drivers (and the simulator) in child processes, for GUIs whose redraws would
    otherwise stall the driver threads.
    """
    cfg = load_config(args.config)
    only = pick_only(args, only)
    sim = None
    ports = {k: v for k, v in (("vesc", args.vesc), ("bno", args.bno)) if v}
    scenario = scenario or args.sim
    if scenario:
        use_sim_paths(cfg)
    if not processes:
        if scenario:
            sim = SimServer(cfg, scenario).start()
            atexit.register(sim.stop)
            ports = sim.ports
        return cfg, make_sources(cfg, ports, only), sim

    from .procs import AcquisitionProcess, SimProcess
    if scenario:
        sim = SimProcess(cfg.path, scenario)
        atexit.register(sim.stop)
        ports = sim.ports
    names = [n for n, ex in cfg.imus.items()
             if (not only or n in only) and (ports.get(n) or (not scenario and ex.extra.get("port")))]
    acq = AcquisitionProcess(cfg.path, ports, names, sim_paths=bool(scenario))
    atexit.register(acq.stop)
    return cfg, list(acq.sources.values()), sim
