import argparse
import asyncio
import os
import signal
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import websockets

# Same binding used by nmmo3.py
from pufferlib.ocean.nmmo3 import binding

# ---- Wire protocol (must match nmmo3.c viewer) ----
# 'NMM3' as big-endian ASCII
NMMO_MAGIC = 0x4E4D4D33

MAP_HDR_FMT = ">IHHiiii"          # uint32, uint16, uint16, int32 x4
TICK_HDR_FMT = ">II"              # uint32 tick, uint32 n_entities
ENT_FMT = ">iiiiiiiiii"           # 10 x int32

# Observation shape used by the env bindings (copied from nmmo3.py)
OBS_SIZE = 11 * 15 * 10 + 47 + 10  # 1707
N_ACTIONS = 26


@dataclass
class EnvConfig:
    width: int = 30
    height: int = 30
    num_players: int = 4
    num_enemies: int = 0
    num_resources: int = 0
    num_weapons: int = 0
    num_gems: int = 0
    tiers: int = 5
    levels: int = 40
    teleportitis_prob: float = 0.0
    enemy_respawn_ticks: int = 2
    item_respawn_ticks: int = 100
    x_window: int = 7
    y_window: int = 5
    reward_combat_level: float = 1.0
    reward_prof_level: float = 1.0
    reward_item_level: float = 0.5
    reward_market: float = 0.01
    reward_death: float = -1.0
    seed: int = 0


class NMMOEnv:
    """Minimal wrapper around the C bindings: one env, many agents."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

        # Arrays that the C env writes into / reads from
        self.obs = np.zeros((cfg.num_players, OBS_SIZE), dtype=np.uint8)
        self.actions = np.zeros((cfg.num_players,), dtype=np.intc)
        self.rewards = np.zeros((cfg.num_players, ), dtype=np.float32)
        self.terminals = np.zeros((cfg.num_players,), dtype=np.bool_)
        self.truncations = np.zeros((cfg.num_players,), dtype=np.bool_)

        self.env_id = binding.env_init(
            self.obs,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            cfg.seed,
            width=cfg.width,
            height=cfg.height,
            num_players=cfg.num_players,
            num_enemies=cfg.num_enemies,
            num_resources=cfg.num_resources,
            num_weapons=cfg.num_weapons,
            num_gems=cfg.num_gems,
            tiers=cfg.tiers,
            levels=cfg.levels,
            teleportitis_prob=cfg.teleportitis_prob,
            enemy_respawn_ticks=cfg.enemy_respawn_ticks,
            item_respawn_ticks=cfg.item_respawn_ticks,
            x_window=cfg.x_window,
            y_window=cfg.y_window,
            reward_combat_level=cfg.reward_combat_level,
            reward_prof_level=cfg.reward_prof_level,
            reward_item_level=cfg.reward_item_level,
            reward_market=cfg.reward_market,
            reward_death=cfg.reward_death,
        )

        self.vec = binding.vectorize(self.env_id)
        binding.vec_reset(self.vec, cfg.seed)

    def step_random(self) -> bool:
        self.actions[:] = np.random.randint(0, N_ACTIONS, size=self.actions.shape, dtype=np.intc)
        binding.vec_step(self.vec)

        if self.terminals.any() or self.truncations.any():
            self.rewards.fill(0)
            binding.vec_reset(self.vec, self.cfg.seed)
            return False  # ← tell caller to SKIP this tick

        return True

    def close(self) -> None:
        try:
            binding.vec_close(self.vec)
        except Exception:
            pass

def build_map_packet(env_handle: int) -> bytes:
    header, terrain = binding.net_map_v1(env_handle)

    width = int(header["width"])
    height = int(header["height"])
    num_players = int(header["num_players"])
    num_enemies = int(header["num_enemies"])

    expected = width * height

    # ---- TERRAIN HANDLING (correct) ----
    if isinstance(terrain, (bytes, bytearray)):
        terrain_bytes = bytes(terrain)

    else:
        terrain_arr = np.asarray(terrain)

        # If it's an object or string array, flatten to raw bytes
        if terrain_arr.dtype.kind in ("O", "S", "U"):
            terrain_bytes = b"".join(terrain_arr.ravel().tolist())
        else:
            # Numeric array
            terrain_bytes = terrain_arr.astype(np.uint8, copy=False).tobytes(order="C")

    if len(terrain_bytes) != expected:
        raise ValueError(
            f"terrain length {len(terrain_bytes)} != expected {expected} "
            f"(type={type(terrain)})"
        )

    hdr = struct.pack(
        MAP_HDR_FMT,
        NMMO_MAGIC,
        1,
        0,
        width,
        height,
        num_players,
        num_enemies,
    )

    return hdr + terrain_bytes



def build_tick_packet(env_handle: int) -> bytes:
    header, entities = binding.net_tick_v1(env_handle)

    entities = list(entities)
    tick = int(header["tick"])

    # binding returns all players+enemies; keep it consistent with the payload.
    n_entities = int(header.get("n_entities", len(entities)))
    if n_entities != len(entities):
        n_entities = len(entities)

    out = bytearray(struct.pack(TICK_HDR_FMT, tick, n_entities))

    for e in entities:
        out += struct.pack(
            ENT_FMT,
            int(e["id"]),
            int(e["type"]),
            int(e["r"]),
            int(e["c"]),
            int(e["hp"]),
            int(e["hp_max"]),
            int(e["comb_lvl"]),
            int(e["element"]),
            int(e["anim"]),
            int(e["dir"]),
        )

    return bytes(out)


class BroadcastHub:
    def __init__(self):
        self._clients = set()
        self.latest_tick = None

    def has_clients(self):
        return bool(self._clients)

    def add(self) -> asyncio.Queue[bytes]:
        q = asyncio.Queue(maxsize=2)  # drop when overloaded
        self._clients.add(q)
        return q

    def remove(self, q: asyncio.Queue[bytes]) -> None:
        self._clients.discard(q)

    def publish(self, payload: bytes) -> None:
        self.latest_tick = payload
        # Called from the asyncio thread (event loop)
        for q in list(self._clients):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop oldest tick to keep latency down
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass


class Producer(threading.Thread):
    def __init__(self, env: NMMOEnv, hub: BroadcastHub, loop: asyncio.AbstractEventLoop, hz: float):
        super().__init__(daemon=True)
        self.env = env
        self.hub = hub
        self.loop = loop
        self.hz = hz
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        period = 1.0 / max(self.hz, 1e-6)
        next_t = time.monotonic()

        while True:
            if self.stop_event.is_set():
                break
            if not self.hub.has_clients():
                time.sleep(0.01)
                continue
            self.env.step_random()
            tick_packet = build_tick_packet(self.env.env_id)

            # Push to asyncio loop
            self.loop.call_soon_threadsafe(self.hub.publish, tick_packet)

            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # If we fell behind, resync
                next_t = time.monotonic()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ws-port", type=int, default=8080)
    ap.add_argument("--tick-hz", type=float, default=1.0)

    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--players", type=int, default=10)
    ap.add_argument("--enemies", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


async def main() -> None:
    args = parse_args()

    cfg = EnvConfig(
        width=args.width,
        height=args.height,
        num_players=args.players,
        num_enemies=args.enemies,
        seed=args.seed,
    )

    env = NMMOEnv(cfg)
    hub = BroadcastHub()

    map_packet = build_map_packet(env.env_id)

    loop = asyncio.get_running_loop()
    producer = Producer(env, hub, loop, hz=args.tick_hz)
    producer.start()

    stop = asyncio.Event()

    def _stop(*_a: Any) -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        pass

    async def tcp_handler(reader: asyncio.StreamReader,
                          writer: asyncio.StreamWriter) -> None:
        q = hub.add()
        peer = writer.get_extra_info("peername")
        try:
            # Bootstrap: header+terrain, then a snapshot tick.
            writer.write(map_packet)
            if hub.latest_tick:
                writer.write(hub.latest_tick)
            await writer.drain()

            while True:
                payload = await q.get()
                writer.write(payload)
                await writer.drain()
        except (asyncio.CancelledError, ConnectionError, BrokenPipeError):
            pass
        finally:
            hub.remove(q)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            if peer:
                print(f"[hi.py] TCP client disconnected: {peer}")

    from websockets.exceptions import ConnectionClosed

    async def ws_handler(ws) -> None:
        q = hub.add()
        try:
            await ws.send(map_packet)
            if hub.latest_tick:
                await ws.send(hub.latest_tick)

            async def drain_incoming():
                async for _ in ws:
                    pass

            drain_task = asyncio.create_task(drain_incoming())
            try:
                while True:
                    payload = await q.get()
                    await ws.send(payload)
            except ConnectionClosed:
                pass
            finally:
                drain_task.cancel()

        finally:
            hub.remove(q)

    ws_server = await websockets.serve(
        ws_handler,
        args.bind,
        args.ws_port,
        max_size=None,
        ping_interval=None,
        compression=None,
        origins=None,
    )
    port = 81
    tcp_server = await asyncio.start_server(
        tcp_handler,
        args.bind,
        port,
    )

    print(f"[hi.py] serving NMMO TCP on {args.bind}:port")
    print(f"[hi.py] serving NMMO websocket on ws://{args.bind}:{args.ws_port}")
    print("[hi.py] connect from the emscripten viewer (nmmo3.c) to that URL")

    try:
        await stop.wait()
    finally:
        ws_server.close()
        await ws_server.wait_closed()

        tcp_server.close()
        await tcp_server.wait_closed()

        producer.stop()
        producer.join()
        env.close()


if __name__ == "__main__":
    if os.getenv("MB_VIEWER") == "true":
        from flask import Flask, send_from_directory

        app = Flask(__name__)


        @app.route("/")
        def index():
            return send_from_directory("games/nmmo3", "game.html")


        if __name__ == "__main__":
            app.run(host="0.0.0.0", port=8080)

    else:
        asyncio.run(main())
