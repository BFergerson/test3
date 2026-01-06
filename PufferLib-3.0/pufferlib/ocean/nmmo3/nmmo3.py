from __future__ import annotations

import argparse
import asyncio
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium
import numpy as np
import pyaudio
import torch
from google import genai
from google.genai import types

import pufferlib
import pufferlib.pufferl as pufferl
# noinspection PyUnresolvedReferences
from pufferlib.ocean.nmmo3 import binding


# -----------------------------
# Environment (same behavior as before)
# -----------------------------
class NMMO3(pufferlib.PufferEnv):
    def __init__(
            self,
            width=8 * [30],
            height=8 * [30],
            num_envs=1,
            num_players=1,
            num_enemies=0,
            num_resources=1,
            num_weapons=0,
            num_gems=0,
            tiers=5,
            levels=40,
            teleportitis_prob=0.0,
            enemy_respawn_ticks=2,
            item_respawn_ticks=100,
            x_window=7,
            y_window=5,
            reward_combat_level=1.0,
            reward_prof_level=1.0,
            reward_item_level=0.5,
            reward_market=0.01,
            reward_death=-1.0,
            log_interval=128,
            buf=None,
            seed=0,
    ):
        self.log_interval = log_interval

        if len(width) > num_envs:
            width = width[:num_envs]
        if len(height) > num_envs:
            height = height[:num_envs]

        if not isinstance(width, list):
            width = num_envs * [width]
        if not isinstance(height, list):
            height = num_envs * [height]
        if not isinstance(num_players, list):
            num_players = num_envs * [num_players]
        if not isinstance(num_enemies, list):
            num_enemies = num_envs * [num_enemies]
        if not isinstance(num_resources, list):
            num_resources = num_envs * [num_resources]
        if not isinstance(num_weapons, list):
            num_weapons = num_envs * [num_weapons]
        if not isinstance(num_gems, list):
            num_gems = num_envs * [num_gems]
        if not isinstance(tiers, list):
            tiers = num_envs * [tiers]
        if not isinstance(levels, list):
            levels = num_envs * [levels]
        if not isinstance(teleportitis_prob, list):
            teleportitis_prob = num_envs * [teleportitis_prob]
        if not isinstance(enemy_respawn_ticks, list):
            enemy_respawn_ticks = num_envs * [enemy_respawn_ticks]
        if not isinstance(item_respawn_ticks, list):
            item_respawn_ticks = num_envs * [item_respawn_ticks]

        assert len(width) == num_envs
        assert len(height) == num_envs

        total_players = 0
        total_enemies = 0
        for idx in range(num_envs):
            if num_players[idx] is None:
                num_players[idx] = width[idx] * height[idx] // 2048
            if num_enemies[idx] is None:
                num_enemies[idx] = width[idx] * height[idx] // 512
            if num_resources[idx] is None:
                num_resources[idx] = width[idx] * height[idx] // 1024
            if num_weapons[idx] is None:
                num_weapons[idx] = width[idx] * height[idx] // 2048
            if num_gems[idx] is None:
                num_gems[idx] = width[idx] * height[idx] // 4096
            if tiers[idx] is None:
                tiers[idx] = 1 if height[idx] <= 128 else 2 if height[idx] <= 256 else 3 if height[idx] <= 512 else 4 if \
                    height[idx] <= 1024 else 5
            if levels[idx] is None:
                levels[idx] = 7 if height[idx] <= 128 else 15 if height[idx] <= 256 else 31 if height[
                                                                                                   idx] <= 512 else 63 if \
                    height[idx] <= 1024 else 99

            total_players += num_players[idx]
            total_enemies += num_enemies[idx]

        self.players_flat = np.zeros((total_players, 51 + 501 + 3), dtype=np.intc)
        self.enemies_flat = np.zeros((total_enemies, 51 + 501 + 3), dtype=np.intc)
        self.rewards_flat = np.zeros((total_players, 10), dtype=np.float32)
        self.actions = np.zeros((total_players,), dtype=np.intc)

        self.num_agents = total_players
        self.num_players = total_players
        self.num_enemies = total_enemies

        self.tick = 0
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(11 * 15 * 10 + 47 + 10,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(26)
        self.render_mode = "human"

        super().__init__(buf)

        player_count = 0
        c_envs = []
        for i in range(num_envs):
            players = num_players[i]
            env_id = binding.env_init(
                self.observations[player_count: player_count + players],
                self.actions[player_count: player_count + players],
                self.rewards[player_count: player_count + players],
                self.terminals[player_count: player_count + players],
                self.truncations[player_count: player_count + players],
                i + seed * num_envs,
                width=width[i],
                height=height[i],
                num_players=num_players[i],
                num_enemies=num_enemies[i],
                num_resources=num_resources[i],
                num_weapons=num_weapons[i],
                num_gems=num_gems[i],
                tiers=tiers[i],
                levels=levels[i],
                teleportitis_prob=teleportitis_prob[i],
                enemy_respawn_ticks=enemy_respawn_ticks[i],
                item_respawn_ticks=item_respawn_ticks[i],
                x_window=x_window,
                y_window=y_window,
                reward_combat_level=reward_combat_level,
                reward_prof_level=reward_prof_level,
                reward_item_level=reward_item_level,
                reward_market=reward_market,
                reward_death=reward_death,
            )
            c_envs.append(env_id)
            player_count += players

        self.c_envs = binding.vectorize(*c_envs)
        self.env_ids = c_envs

    def reset(self, seed=0):
        self.rewards.fill(0)
        self.is_reset = True
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        if not hasattr(self, "is_reset"):
            raise RuntimeError("Must call reset before step")
        self.rewards.fill(0)
        self.actions[:] = actions[:]
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        self.tick += 1
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        for _ in range(36):
            binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


# -----------------------------
# Policy loading + act
# -----------------------------
class DummyVecEnv:
    def __init__(self, driver_env):
        self.driver_env = driver_env


def load_policy_from_pt(env, env_name: str, ckpt_path: str):
    args = pufferl.load_config(env_name)
    args["load_model_path"] = ckpt_path
    vec = DummyVecEnv(env)
    policy = pufferl.load_policy(args, vec, env_name=env_name)
    policy.eval()
    return policy, args


@torch.inference_mode()
def policy_act(policy, obs_np, reward_np, done_np, device, use_rnn=False, lstm_state=None):
    obs = torch.as_tensor(obs_np, device=device)
    state = dict(
        reward=torch.as_tensor(reward_np, device=device),
        done=torch.as_tensor(done_np, device=device),
        env_id=slice(0, obs.shape[0]),
        mask=np.ones(obs.shape[0], dtype=np.bool_),
    )

    if use_rnn:
        if lstm_state is None:
            hsize = policy.hidden_size
            lstm_state = (
                torch.zeros(obs.shape[0], hsize, device=device),
                torch.zeros(obs.shape[0], hsize, device=device),
            )
        state["lstm_h"], state["lstm_c"] = lstm_state

    logits, _value = policy.forward_eval(obs, state)
    action, _logprob, _ = pufferlib.pytorch.sample_logits(logits)
    action_np = action.detach().cpu().numpy().astype(np.int32)

    if use_rnn:
        return action_np, (state["lstm_h"], state["lstm_c"])
    return action_np, None


# -----------------------------
# Thread-safe swap flag
# -----------------------------
class AtomicSwap:
    def __init__(self, initial_swapped: bool = False):
        self._lock = threading.Lock()
        self._swapped = initial_swapped

    def toggle(self) -> bool:
        with self._lock:
            self._swapped = not self._swapped
            return self._swapped

    def get(self) -> bool:
        with self._lock:
            return self._swapped


# -----------------------------
# Game thread: env + policy inference isolated from asyncio + audio
# -----------------------------
@dataclass
class GameConfig:
    left_pt: str
    right_pt: str
    env_name: str
    render_every: int = 6
    timeout_s: Optional[float] = None
    clamp_torch_threads: bool = True

def _tile_to_char(tile: int) -> str:
    """
    Extremely low-detail terrain symbol.
    Based on the TILE_* constants in nmmo3.h:
      0–3   grass
      4–7   dirt
      8–11  stone
      12–15 water
    Everything else collapses to "?".
    """
    if 12 <= tile <= 15:
        return "~"   # water
    if 8 <= tile <= 11:
        return "^"   # stone / mountain
    if 4 <= tile <= 7:
        return ","   # dirt
    if 0 <= tile <= 3:
        return "."   # grass
    return "?"       # unknown / other


def ascii_minimap(env_handle: int, max_cols: int = 64, max_rows: int = 32) -> None:
    """
    Print an extremely low-detail picture of the game world.

    - env_handle: C env handle (same thing you pass to net_*_v1)
    - max_cols / max_rows: max size of the printed minimap
    """
    # Get map + tick snapshots from the C bindings
    map_header, terrain = binding.net_map_v1(env_handle)
    tick_header, entities = binding.net_tick_v1(env_handle)

    W = map_header["width"]
    H = map_header["height"]
    num_players = map_header["num_players"]

    # Downsample factors (at least 1)
    sx = max(1, W // max_cols)
    sy = max(1, H // max_rows)

    out_w = (W + sx - 1) // sx
    out_h = (H + sy - 1) // sy

    # Initialize grid from terrain
    grid = [[" " for _ in range(out_w)] for _ in range(out_h)]

    for oy in range(out_h):
        r0 = min(oy * sy, H - 1)
        for ox in range(out_w):
            c0 = min(ox * sx, W - 1)
            tile = terrain[r0 * W + c0]  # bytes -> int
            grid[oy][ox] = _tile_to_char(tile)

    # Overlay entities (players then enemies)
    # id < num_players => player, else enemy
    for e in entities:
        r = e["r"]
        c = e["c"]
        if not (0 <= r < H and 0 <= c < W):
            continue

        oy = r // sy
        ox = c // sx
        if not (0 <= oy < out_h and 0 <= ox < out_w):
            continue

        symbol = "P" if e["id"] < num_players else "M"
        current = grid[oy][ox]

        # Priority: P > M > terrain
        if symbol == "P":
            grid[oy][ox] = "P"
        elif current not in ("P", "M"):
            grid[oy][ox] = "M"

    # Print it
    print(f"=== NMMO MINIMAP (tick {tick_header['tick']}) ===")
    print(f"world {W}x{H} -> {out_w}x{out_h} (sx={sx}, sy={sy})")
    for oy in range(out_h):
        print("".join(grid[oy]))

def ascii_minimap_env(env, idx: int = 0, **kwargs) -> None:
    """Render minimap for a Python NMMO3 env (by index in env.env_ids)."""
    ascii_minimap(env.env_ids[idx], **kwargs)


class GameRunner(threading.Thread):
    def __init__(self, cfg: GameConfig, swap: AtomicSwap):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.swap = swap
        self.stop_event = threading.Event()

        self._tick = 0
        self._sps_last_t = 0.0
        self._sps_last_tick = 0

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        if self.cfg.clamp_torch_threads:
            # Prevent Torch from competing with audio/network threads on CPU.
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        # EXACTLY two agents in a single env
        env = NMMO3(num_players=2)
        obs, _ = env.reset()

        # Grab the first C env handle
        h = env.env_ids[0]

        # Print map once
        nmmo_send_map_v1(h)

        if env.num_agents != 2:
            raise RuntimeError(f"Expected exactly 2 agents, got {env.num_agents}. "
                               f"Check NMMO3(num_players=2) wiring.")

        left_policy, args = load_policy_from_pt(env, self.cfg.env_name, self.cfg.left_pt)
        right_policy, _ = load_policy_from_pt(env, self.cfg.env_name, self.cfg.right_pt)

        device = args["train"]["device"]
        use_rnn = args["train"]["use_rnn"]

        reward = np.zeros(env.num_agents, dtype=np.float32)
        done = np.zeros(env.num_agents, dtype=np.bool_)

        # Keep RNN state per (agent_idx, policy_name) so swapping doesn't corrupt hidden state.
        lstm = {
            (0, "left"): None,
            (0, "right"): None,
            (1, "left"): None,
            (1, "right"): None,
        }

        start = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if self.cfg.timeout_s is not None and (time.monotonic() - start) >= self.cfg.timeout_s:
                    break

                swapped = self.swap.get()

                # Normal:
                #   agent0 -> left_policy
                #   agent1 -> right_policy
                # Swapped:
                #   agent0 -> right_policy
                #   agent1 -> left_policy
                a0_pol = "right" if swapped else "left"
                a1_pol = "left" if swapped else "right"

                atn = np.empty(env.num_agents, dtype=np.int32)

                # Agent 0
                act0, lstm[(0, a0_pol)] = policy_act(
                    left_policy if a0_pol == "left" else right_policy,
                    obs[0:1],
                    reward[0:1],
                    done[0:1],
                    device,
                    use_rnn,
                    lstm[(0, a0_pol)],
                )
                atn[0] = int(act0[0])

                # Agent 1
                act1, lstm[(1, a1_pol)] = policy_act(
                    left_policy if a1_pol == "left" else right_policy,
                    obs[1:2],
                    reward[1:2],
                    done[1:2],
                    device,
                    use_rnn,
                    lstm[(1, a1_pol)],
                )
                atn[1] = int(act1[0])

                # env step
                obs, reward, terminals, truncations, _info = env.step(atn)
                done = np.asarray(terminals, dtype=np.bool_) | np.asarray(truncations, dtype=np.bool_)
                # nmmo_send_tick_v1(h)

                # if self._tick % 1 == 0:
                #     ascii_minimap_env(env, max_cols=30, max_rows=30)

                # render (optional, heavy)
                if self.cfg.render_every > 0 and (self._tick % self.cfg.render_every == 0):
                    env.render()

                if done.any():
                    obs, _ = env.reset()
                    reward[:] = 0
                    done[:] = False
                    for k in lstm:
                        lstm[k] = None

                self._tick += 1

                # lightweight SPS log every ~2s
                now = time.monotonic()
                if now - self._sps_last_t >= 2.0:
                    dt = max(now - self._sps_last_t, 1e-6)
                    d_ticks = self._tick - self._sps_last_tick
                    sps = env.num_agents * d_ticks / dt
                    print(
                        f"[GAME] swapped={int(swapped)} "
                        f"A0={a0_pol:<5} A1={a1_pol:<5} "
                        f"tick={self._tick:<8} SPS={sps:,.0f}"
                    )
                    self._sps_last_t = now
                    self._sps_last_tick = self._tick
        finally:
            env.close()


# -----------------------------
# Audio threads: blocking PyAudio I/O (no per-chunk asyncio.to_thread)
# -----------------------------
@dataclass
class AudioConfig:
    send_rate: int = 16000
    recv_rate: int = 24000
    chunk: int = 1024
    channels: int = 1
    fmt = pyaudio.paInt16

    mic_queue_max: int = 32  # bounded -> drop when overloaded
    spk_queue_max: int = 64  # bounded -> drop when overloaded


class AudioIO:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.pya = pyaudio.PyAudio()

        self.mic_q: queue.Queue[bytes] = queue.Queue(maxsize=cfg.mic_queue_max)
        self.spk_q: queue.Queue[bytes] = queue.Queue(maxsize=cfg.spk_queue_max)

        self._stop = threading.Event()
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._spk_thread = threading.Thread(target=self._spk_loop, daemon=True)

        self._mic_stream = None
        self._spk_stream = None

    def start(self) -> None:
        self._mic_thread.start()
        self._spk_thread.start()

    def stop(self) -> None:
        self._stop.set()

        # Unblock mic sender if it's waiting on mic_q.get()
        try:
            self.mic_q.put_nowait(b"")
        except queue.Full:
            # Drop one item then retry
            try:
                _ = self.mic_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.mic_q.put_nowait(b"")
            except queue.Full:
                pass

        # Unblock speaker thread if it's waiting
        try:
            self.spk_q.put_nowait(b"")
        except queue.Full:
            pass

        self._mic_thread.join(timeout=2)
        self._spk_thread.join(timeout=2)

        if self._mic_stream is not None:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
        if self._spk_stream is not None:
            self._spk_stream.stop_stream()
            self._spk_stream.close()

        self.pya.terminate()

    def _mic_loop(self) -> None:
        mic_info = self.pya.get_default_input_device_info()
        self._mic_stream = self.pya.open(
            format=self.cfg.fmt,
            channels=self.cfg.channels,
            rate=self.cfg.send_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.cfg.chunk,
        )
        kwargs = {"exception_on_overflow": False}

        while not self._stop.is_set():
            data = self._mic_stream.read(self.cfg.chunk, **kwargs)

            # Drop policy: keep realtime by dropping newest if queue is full
            try:
                self.mic_q.put_nowait(data)
            except queue.Full:
                # Drop oldest to keep latency down
                try:
                    _ = self.mic_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.mic_q.put_nowait(data)
                except queue.Full:
                    pass

    def _spk_loop(self) -> None:
        self._spk_stream = self.pya.open(
            format=self.cfg.fmt,
            channels=self.cfg.channels,
            rate=self.cfg.recv_rate,
            output=True,
        )
        while not self._stop.is_set():
            data = self.spk_q.get()
            if not data:
                continue
            self._spk_stream.write(data)


# -----------------------------
# Gemini Live
# -----------------------------
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


def build_tools():
    switch_decl = {
        "name": "switch",
        "description": (
            "Swap which agent uses which policy. "
            "Call this ONLY when the user clearly says 'switch'."
        ),
    }
    return [{"function_declarations": [switch_decl]}]


def build_config():
    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": (
            "You are a helpful assistant. "
            "You can control one tool: switch. "
            "Only invoke switch when the user explicitly says 'switch'. "
            "Do NOT call this tool for any other request."
        ),
        "input_audio_transcription": {},
        "tools": build_tools(),
    }


async def mic_sender(live_session, audio: AudioIO):
    """
    Bridge mic thread -> Gemini (mirrors gemini_live.py).

    Reads raw 16kHz PCM chunks from the mic thread queue and forwards them to Gemini.
    Uses a shutdown sentinel (b"") so we don't strand a blocking queue.get() in a worker
    thread during teardown.
    """
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await loop.run_in_executor(None, audio.mic_q.get)
            if not data:
                return  # shutdown sentinel
            await live_session.send_realtime_input(
                audio={
                    "data": data,
                    "mime_type": "audio/pcm;rate=16000",
                }
            )
    except asyncio.CancelledError:
        return


async def gemini_receiver(live_session, audio: AudioIO, swap: AtomicSwap):
    """
    Mirrors gemini_live.py receive_and_handle:

    - Streams TTS audio (24kHz PCM bytes) into the speaker queue.
    - Executes tool call (switch) by toggling the policy assignment.
    """
    try:
        while True:
            async for response in live_session.receive():
                # 1) Audio data for TTS response
                if getattr(response, "data", None) is not None:
                    try:
                        audio.spk_q.put_nowait(response.data)
                    except queue.Full:
                        # Drop oldest to keep latency down
                        try:
                            _ = audio.spk_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            audio.spk_q.put_nowait(response.data)
                        except queue.Full:
                            pass

                # 2) Tool calls (function calling)
                elif getattr(response, "tool_call", None):
                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        if fc.name == "switch":
                            swapped = swap.toggle()
                            a0 = "right" if swapped else "left"
                            a1 = "left" if swapped else "right"
                            print(f"[SWITCH] swapped={int(swapped)} A0={a0} A1={a1}")

                        function_responses.append(
                            types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={"result": "ok"},
                            )
                        )

                    await live_session.send_tool_response(
                        function_responses=function_responses
                    )
    except asyncio.CancelledError:
        return


# -----------------------------
# Entrypoint
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-pt", required=False, default="left.pt")
    ap.add_argument("--right-pt", required=False, default="right.pt")
    ap.add_argument("--env-name", default="puffer_nmmo3")
    ap.add_argument("--render-every", type=int, default=1, help="0 to disable render")
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--no-clamp-torch-threads", action="store_true")
    return ap.parse_args()

# Thin Python versions of the C networking functions that just print.

def nmmo_send_map_v1(env_handle: int) -> None:
    """
    Debug replacement for the C nmmo_send_map_v1.
    Instead of writing to a socket, it prints a summary to stdout.
    """
    header, terrain = binding.net_map_v1(env_handle)

    print("=== NMMO MAP V1 ===")
    print(f"magic=0x{header['magic']:08X} version={header['version']}")
    print(
        f"size={header['width']}x{header['height']} "
        f"players={header['num_players']} enemies={header['num_enemies']}"
    )
    print(f"terrain_bytes={len(terrain)}")
    # If you want a little visual sample:
    w = header["width"]
    h = header["height"]
    sample_h = min(h, 5)
    print("terrain sample:")
    for r in range(sample_h):
        row = terrain[r * w:(r + 1) * w]
        # Print as integers (tile ids)
        print(" ", " ".join(f"{b:02X}" for b in row))


def nmmo_send_tick_v1(env_handle: int) -> None:
    """
    Debug replacement for the C nmmo_send_tick_v1.
    Prints the current tick header + entities to stdout.
    """
    header, entities = binding.net_tick_v1(env_handle)

    print("=== NMMO TICK V1 ===")
    print(f"tick={header['tick']} n_entities={header['n_entities']}")

    # Basic dump: one line per entity
    for e in entities:
        print(
            f"id={e['id']:3d} "
            f"type={e['type']:2d} "
            f"pos=({e['r']:3d},{e['c']:3d}) "
            f"hp={e['hp']:3d}/{e['hp_max']:3d} "
            f"comb={e['comb_lvl']:2d} "
            f"elem={e['element']:2d} "
            f"anim={e['anim']:2d} "
            f"dir={e['dir']:2d}"
        )


async def main_async():
    print(binding.hello_world())
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    swap = AtomicSwap(False)

    game_cfg = GameConfig(
        left_pt=args.left_pt,
        right_pt=args.right_pt,
        env_name=args.env_name,
        render_every=args.render_every,
        timeout_s=args.timeout_s,
        clamp_torch_threads=not args.no_clamp_torch_threads,
    )
    game = GameRunner(game_cfg, swap)

    audio = AudioIO(AudioConfig())
    audio.start()
    game.start()

    client = genai.Client(api_key=api_key)
    config = build_config()

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as live_session:
            print("Connected to Gemini. Say 'switch' to swap agent policies.")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(mic_sender(live_session, audio))
                tg.create_task(gemini_receiver(live_session, audio, swap))
    finally:
        game.stop()
        audio.stop()
        game.join(timeout=2)
        print("\nShutdown complete.")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
