"""Run the serial drivers (and the simulator) in separate processes.

Python threads share one interpreter lock, so a busy GUI (matplotlib redraws take tens of ms) delays the
driver threads: VESC polls stall and, worse, request/reply timestamps get stretched. Running acquisition
in its own process keeps sampling and timestamps clean. CLOCK_MONOTONIC is system-wide on Linux, so
timestamps taken in the worker process are directly comparable with the GUI process.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time

from .frames import load_config
from .pipeline import make_sources


def _acq_worker(cfg_path, sim_paths, ports, only, q_out, stop_evt, q_cmd=None):
    cfg = load_config(cfg_path)
    if sim_paths:
        from .cli import use_sim_paths
        use_sim_paths(cfg)
    sources = make_sources(cfg, ports, only)
    for s in sources:
        s.start()
    last_stats = 0.0
    by_name = {s.src_name: s for s in sources}
    while not stop_evt.is_set():
        time.sleep(0.005)
        while q_cmd is not None:                 # drive commands from the GUI process
            try:
                name, method, kwargs = q_cmd.get_nowait()
            except queue.Empty:
                break
            src = by_name.get(name)
            if src is not None and hasattr(src, method):
                getattr(src, method)(**kwargs)
        batch = [it for s in sources for it in s.drain()]
        if batch:
            q_out.put(("data", batch))
        now = time.monotonic()
        if now - last_stats > 0.5:
            last_stats = now
            q_out.put(("stats", {s.src_name: {"rate_hz": s.rate_hz, "errors": s.errors, "last_error": s.last_error,
                                              "count": s.count, "port": getattr(s, "port", ""),
                                              "drive_sent": getattr(s, "drive_sent", 0),
                                              "rtt": s.rtt_stats() if hasattr(s, "rtt_stats") else {}}
                                 for s in sources}))
    for s in sources:
        s.stop()
    for s in sources:
        s.join(1.0)


class RemoteSource:
    """Looks like a racer_imu.source.Source to Pipeline, but the driver runs in the worker process."""

    def __init__(self, group: "AcquisitionProcess", name: str):
        self.group, self.src_name = group, name
        self.items: list = []
        self.rate_hz, self.errors, self.last_error, self.count, self.port = 0.0, 0, "", 0, ""
        self.drive_sent = 0
        self._rtt: dict = {}

    def start(self):
        self.group.start()

    def stop(self):
        self.group.stop()

    def join(self, timeout=None):
        self.group.join(timeout)

    def drain(self):
        self.group.pump()
        out, self.items = self.items, []
        return out

    def rtt_stats(self):
        return self._rtt

    def set_drive(self, **kwargs):
        self.group.q_cmd.put((self.src_name, "set_drive", kwargs))

    def stop_drive(self):
        self.group.q_cmd.put((self.src_name, "stop_drive", {}))


class AcquisitionProcess:
    def __init__(self, cfg_path, ports: dict, names: list[str], sim_paths: bool = False):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.q_cmd = ctx.Queue()
        self.stop_evt = ctx.Event()
        self.proc = ctx.Process(target=_acq_worker,
                                args=(str(cfg_path), sim_paths, ports, names, self.q, self.stop_evt, self.q_cmd),
                                daemon=True)
        self.sources = {n: RemoteSource(self, n) for n in names}
        self._started = False

    def start(self):
        if not self._started:
            self._started = True
            self.proc.start()

    def pump(self):
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                return
            if kind == "data":
                for it in payload:
                    src = self.sources.get(getattr(it, "src", None))
                    if src is not None:
                        src.items.append(it)
            else:
                for n, st in payload.items():
                    if n in self.sources:
                        s = self.sources[n]
                        s.rate_hz, s.errors, s.last_error = st["rate_hz"], st["errors"], st["last_error"]
                        s.count, s.port, s._rtt = st["count"], st["port"], st["rtt"]
                        s.drive_sent = st.get("drive_sent", 0)

    def stop(self):
        self.stop_evt.set()

    def join(self, timeout=None):
        if self.proc.is_alive():
            self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()


# ---------------------------------------------------------------- simulator in its own process
def _sim_worker(cfg_path, scenario, seed, conn, stop_evt):
    from .sim import SimServer
    srv = SimServer(load_config(cfg_path), scenario, seed).start()
    conn.send({"ports": srv.ports, "t0": srv.t0, "describe": srv.describe()})
    while not stop_evt.is_set():
        time.sleep(0.1)
    srv.stop()


class SimProcess:
    """SimServer in a child process. truth_at() is computed locally from the same trajectory + start time."""

    def __init__(self, cfg_path, scenario: str, seed: int = 0):
        from .sim import Trajectory, default_truth
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.stop_evt = ctx.Event()
        self.proc = ctx.Process(target=_sim_worker, args=(str(cfg_path), scenario, seed, child, self.stop_evt),
                                daemon=True)
        self.proc.start()
        if not parent.poll(30):
            raise RuntimeError("simulator process did not start")
        info = parent.recv()
        self.ports, self.t0, self.info = info["ports"], info["t0"], info["describe"]
        self.traj = Trajectory(scenario, seed=seed)
        self.truth_sensors = default_truth(load_config(cfg_path))

    def truth_at(self, t_host: float) -> dict:
        return self.traj.truth(t_host - self.t0)

    def stop(self):
        self.stop_evt.set()
        self.proc.join(2)
        if self.proc.is_alive():
            self.proc.terminate()
