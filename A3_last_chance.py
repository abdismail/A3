"""
A3_final_learning.py — Assignment 3 (Robot Olympics): GA with true brain evolution
---------------------------------------------------------------------------------
- Evolves BOTH morphology (NDE input vectors) and controller weights (MLP).
- Fitness = forward X progress (start_x -> end_x) with non-learner penalty.
- DeepSeek-style saving: per-gen best JSONs, checkpoints, final controller+robot JSON,
  evolution plot, and optional champion video.
- Opens MuJoCo viewer for the final champion by default (launcher mode).

CLI:
  python A3_final_learning.py
  python A3_final_learning.py --generations 500 --pop 24 --duration 25 --mode launcher --record
"""
from __future__ import annotations
# ----------------------------- USER DEFAULTS ------------------------------
GENERATIONS_DEFAULT = 50        # quick test; use --generations 500 for long run
POP_SIZE_DEFAULT    = 16
DURATION_DEFAULT    = 60        # seconds per simulation
RUN_MODE_DEFAULT    = "launcher"  # "simple" | "video" | "launcher" | "frame" | "no_control"
RECORD_VIDEO_BEST_DEFAULT = False  # set via --record

# ----------------------------- Imports -----------------------------------

from typing import TYPE_CHECKING, Any, Literal, List, Tuple, Dict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
import json
import pickle
import argparse

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer

# ARIEL imports
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from networkx import DiGraph

# ----------------------------- Globals -----------------------------------
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# Arena and target
SPAWN_POS = [-0.8, 0.0, 0.35]        # slightly higher to avoid ground clipping
TARGET_POSITION = [5.0, 0.0, 0.5]

# NDE/morphology parameters
NUM_OF_MODULES = 8
GENOTYPE_SIZE = 64

# Controller architecture
HIDDEN_SIZE = 8   # fixed like template

# Controller scale clamp (kept moderate to avoid QACC explosions)
CTRL_SCALE_RANGE = (0.15 * math.pi, 1.25 * math.pi)

# Non-learner threshold (meters)
NONLEARNER_MIN_FWD = 0.05
NONLEARNER_PENALTY = -10.0


# --------------------------- Utilities ------------------------------------
def forward_progress(history: List[Tuple[float, float, float]]) -> float:
    if not history:
        return -1e9
    start_x = history[0][0]
    end_x   = history[-1][0]
    return float(end_x - start_x)


def fitness_function(history):
    if not history:
        return -100.0
    xs = [p[0] for p in history]
    zs = [p[2] for p in history]
    forward = xs[-1] - xs[0]
    vertical_motion = np.std(zs)
    motion_amount = np.std(xs)
    fitness = 3.0 * forward + 0.5 * motion_amount - 0.2 * vertical_motion
    if forward < 0.01 and motion_amount < 0.01:
        fitness -= 10.0
    return float(fitness)




def show_xpos_history(history: List[Tuple[float, float, float]]) -> None:
    if not history:
        console.log("[show_xpos_history] No history to plot; skipping.")
        return

    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(model, data, camera=camera, save_path=save_path, save=True)

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history, dtype=float)

    # Pixel anchors measured for provided background
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0)) if (yc - y0) != 0 else 1.0
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.show()


# --------------------------- Genome ---------------------------------------
@dataclass
class Genome:
    # Morphology (NDE input vectors)
    type_p: np.ndarray  # (GENOTYPE_SIZE,) in [0,1]
    conn_p: np.ndarray  # (GENOTYPE_SIZE,) in [0,1]
    rot_p:  np.ndarray  # (GENOTYPE_SIZE,) in [0,1]
    # Controller global scale
    ctrl_scale: float   # in CTRL_SCALE_RANGE
    # Controller weights (evolved)
    w1: np.ndarray | None = None  # shape: (input, HIDDEN_SIZE)
    w2: np.ndarray | None = None  # shape: (HIDDEN_SIZE, HIDDEN_SIZE)
    w3: np.ndarray | None = None  # shape: (HIDDEN_SIZE, output)

    # For first-time shape discovery; stored for checkpoints but not strictly required
    input_size: int = 0
    output_size: int = 0

    @staticmethod
    def random() -> "Genome":
        rng = np.random.default_rng()
        return Genome(
            type_p=rng.random(GENOTYPE_SIZE, dtype=np.float32),
            conn_p=rng.random(GENOTYPE_SIZE, dtype=np.float32),
            rot_p= rng.random(GENOTYPE_SIZE, dtype=np.float32),
            ctrl_scale=float(rng.uniform(*CTRL_SCALE_RANGE)),
            w1=None, w2=None, w3=None,
        )

    def copy(self) -> "Genome":
        return Genome(
            self.type_p.copy(),
            self.conn_p.copy(),
            self.rot_p.copy(),
            float(self.ctrl_scale),
            None if self.w1 is None else self.w1.copy(),
            None if self.w2 is None else self.w2.copy(),
            None if self.w3 is None else self.w3.copy(),
            self.input_size,
            self.output_size,
        )


# --------------------------- Decode & Controller --------------------------
def decode_robot(genome: Genome) -> Tuple[Any, "DiGraph[Any]"]:
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward([genome.type_p, genome.conn_p, genome.rot_p])

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: "DiGraph[Any]" = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )
    core = construct_mjspec_from_graph(robot_graph)
    return core, robot_graph


def ensure_weights(genome: Genome, input_size: int, output_size: int) -> None:
    """Initialize controller weights for a genome if they are None (first evaluation)."""
    if genome.w1 is None or genome.w2 is None or genome.w3 is None:
        rng = np.random.default_rng()
        genome.w1 = rng.normal(0.0, 0.5, size=(input_size, HIDDEN_SIZE)).astype(np.float32)
        genome.w2 = rng.normal(0.0, 0.5, size=(HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32)
        genome.w3 = rng.normal(0.0, 0.5, size=(HIDDEN_SIZE, output_size)).astype(np.float32)
        genome.input_size = input_size
        genome.output_size = output_size
    else:
        # If shapes changed (unlikely), re-init safely
        if genome.w1.shape != (input_size, HIDDEN_SIZE) or \
           genome.w2.shape != (HIDDEN_SIZE, HIDDEN_SIZE) or \
           genome.w3.shape != (HIDDEN_SIZE, output_size):
            rng = np.random.default_rng()
            genome.w1 = rng.normal(0.0, 0.5, size=(input_size, HIDDEN_SIZE)).astype(np.float32)
            genome.w2 = rng.normal(0.0, 0.5, size=(HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32)
            genome.w3 = rng.normal(0.0, 0.5, size=(HIDDEN_SIZE, output_size)).astype(np.float32)
            genome.input_size = input_size
            genome.output_size = output_size


def make_controller_from_genome(genome: Genome) -> Controller:
    """Create a Controller whose callback uses the genome's weights directly."""
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    def callback(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        input_size = len(data.qpos)
        output_size = model.nu
        ensure_weights(genome, input_size, output_size)

        w1, w2, w3 = genome.w1, genome.w2, genome.w3
        assert w1 is not None and w2 is not None and w3 is not None

        # Inputs are qpos (like template)
        inputs = data.qpos

        # Forward pass
        layer1 = np.tanh(np.dot(inputs, w1))
        layer2 = np.tanh(np.dot(layer1, w2))
        outputs = np.tanh(np.dot(layer2, w3))
        # Controller output (evolved neural net)
        ctrl = outputs * genome.ctrl_scale

        # --- Movement-starter baseline (identical for all robots) ---
        # Adds a small sinusoidal wave to break symmetry and give minimal motion
        t = data.time
        baseline = 0.6 * np.sin(2.0 * np.pi * 0.5 * t + np.arange(model.nu))
        ctrl = ctrl + baseline  # combine learned output + baseline
        # ------------------------------------------------------------

        # Clip to actuator control range (safe values)
        nu = model.nu
        if nu > 0:
            lo = np.empty(nu)
            hi = np.empty(nu)
            for i in range(nu):
                lo[i] = model.actuator_ctrlrange[i, 0] if model.actuator_ctrlrange is not None else -1.0
                hi[i] = model.actuator_ctrlrange[i, 1] if model.actuator_ctrlrange is not None else 1.0
            ctrl = np.clip(ctrl, lo, hi)

        return ctrl.astype(np.float64)

    return Controller(controller_callback_function=callback, tracker=tracker)


# --------------------------- Evaluation -----------------------------------
def evaluate(genome: Genome, duration: int, mode: ViewerTypes = "simple",
             video_filename: str | None = None) -> Tuple[float, Dict[str, Any]]:
    """Build, run, and score one individual; optionally record a video."""
    # Clear previous callbacks (common pitfall)
    mj.set_mjcb_control(None)

    # World & robot
    world = OlympicArena()
    core, robot_graph = decode_robot(genome)

    # Adaptive spawn height (avoid clipping with larger bodies)
    approx_size = 0.02 * GENOTYPE_SIZE  # heuristic
    spawn_z = max(SPAWN_POS[2], 0.25 + 0.001 * approx_size)
    world.spawn(core.spec, spawn_position=[SPAWN_POS[0], SPAWN_POS[1], spawn_z])

    # Model/data
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state
    mj.mj_resetData(model, data)

    # Controller built from genome weights
    ctrl = make_controller_from_genome(genome)
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    # Register callback
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Run simulation
    if mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "video":
        assert video_filename is not None, "video_filename required for mode='video'"
        video_dir = DATA / "videos"
        video_dir.mkdir(exist_ok=True, parents=True)
        video_recorder = VideoRecorder(output_folder=str(video_dir), filename=video_filename)
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
    elif mode == "launcher":
        try:
            viewer.launch(model=model, data=data)
        except Exception as e:
            console.log(f"[viewer] launch failed: {e!r}. Falling back to simple runner.")
            simple_runner(model, data, duration=duration)
    elif mode == "frame":
        single_frame_renderer(model, data, save=True, save_path=str(DATA / "robot_frame.png"))
    else:
        simple_runner(model, data, duration=duration)

    # Score
    history = []
    if ctrl.tracker is not None and "xpos" in ctrl.tracker.history and len(ctrl.tracker.history["xpos"]) > 0:
        history = ctrl.tracker.history["xpos"][0]
    fit = fitness_function(history)

    return fit, {"history": history, "robot_graph": robot_graph}


# --- GA operators (blend crossover + gaussian mutation + elitism) ---------
def blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.random.default_rng().uniform(-alpha, 1 + alpha, size=a.shape).astype(np.float32)
    c1 = lam * a + (1 - lam) * b
    c2 = lam * b + (1 - lam) * a
    return c1, c2


def crossover(pa: Genome, pb: Genome) -> Tuple[Genome, Genome]:
    c1, c2 = pa.copy(), pb.copy()
    # morphology
    c1.type_p, c2.type_p = blend(pa.type_p, pb.type_p)
    c1.conn_p, c2.conn_p = blend(pa.conn_p, pb.conn_p)
    c1.rot_p,  c2.rot_p  = blend(pa.rot_p,  pb.rot_p)
    # controller scale (arithmetic blend + clip)
    avg_scale = 0.5 * (pa.ctrl_scale + pb.ctrl_scale)
    c1.ctrl_scale = float(np.clip(avg_scale, *CTRL_SCALE_RANGE))
    c2.ctrl_scale = c1.ctrl_scale
    # controller weights (if available): arithmetic blend
    if pa.w1 is not None and pb.w1 is not None and pa.w1.shape == pb.w1.shape:
        c1.w1 = 0.5 * (pa.w1 + pb.w1)
        c2.w1 = c1.w1.copy()
    else:
        c1.w1 = None; c2.w1 = None

    if pa.w2 is not None and pb.w2 is not None and pa.w2.shape == pb.w2.shape:
        c1.w2 = 0.5 * (pa.w2 + pb.w2)
        c2.w2 = c1.w2.copy()
    else:
        c1.w2 = None; c2.w2 = None

    if pa.w3 is not None and pb.w3 is not None and pa.w3.shape == pb.w3.shape:
        c1.w3 = 0.5 * (pa.w3 + pb.w3)
        c2.w3 = c1.w3.copy()
    else:
        c1.w3 = None; c2.w3 = None

    return c1, c2


def mutate_weights(w: np.ndarray, sigma: float, p: float) -> np.ndarray:
    rng = np.random.default_rng()
    if w is None:
        return w
    mask = rng.random(w.shape) < p
    w = w.copy()
    w[mask] += rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
    # optional clip to keep stable
    np.clip(w, -2.5, 2.5, out=w)
    return w


def mutate(g: Genome, sigma: float = 0.05, p: float = 0.2) -> Genome:
    out = g.copy()
    rng = np.random.default_rng()
    # mutate morphology
    for arr in [out.type_p, out.conn_p, out.rot_p]:
        mask = rng.random(arr.shape) < p
        arr[mask] += rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
        np.clip(arr, 0.0, 1.0, out=arr)

    # mutate controller scale
    if rng.random() < 0.6:
        out.ctrl_scale = float(np.clip(out.ctrl_scale + float(rng.normal(0.0, 0.15*math.pi)), *CTRL_SCALE_RANGE))

    # mutate controller weights (if initialized)
    out.w1 = mutate_weights(out.w1, sigma=0.10, p=0.25)
    out.w2 = mutate_weights(out.w2, sigma=0.10, p=0.25)
    out.w3 = mutate_weights(out.w3, sigma=0.10, p=0.25)
    return out


# --------------------------- Saving ---------------------------------------
def save_checkpoint(population: List[Genome], best_genome: Genome, generation_idx: int,
                    fitness_list: List[float], avg_fit: float, best_fit: float) -> None:
    """Save a pickle checkpoint for the current generation."""
    try:
        checkpoint = {
            "generation": generation_idx,
            "population": population,
            "best_genome": best_genome,
            "fitness": fitness_list,
            "avg_fitness": avg_fit,
            "best_fitness": best_fit,
        }
        with open(DATA / f"checkpoint_gen_{generation_idx:03d}.pkl", "wb") as f:
            pickle.dump(checkpoint, f)
    except Exception as e:
        console.log(f"[WARN] Could not save checkpoint for gen {generation_idx}: {e!r}")


def save_best_robot(robot_graph: "DiGraph[Any]", filename: str) -> None:
    """Save robot graph JSON."""
    try:
        save_graph_as_json(robot_graph, DATA / filename)
    except Exception as e:
        console.log(f"[WARN] Saving robot graph failed ({filename}): {e!r}")


def save_final_controller(genome: Genome, filename: str = "controller_best_robot_final.json") -> None:
    """Save evolved controller weights + shapes to JSON (meets 'weights+architecture' requirement)."""
    ctrl = {
        "hidden_size": HIDDEN_SIZE,
        "ctrl_scale": float(genome.ctrl_scale),
        "input_size": int(genome.input_size),
        "output_size": int(genome.output_size),
        # store weights as lists (JSON-friendly)
        "w1": None if genome.w1 is None else genome.w1.tolist(),
        "w2": None if genome.w2 is None else genome.w2.tolist(),
        "w3": None if genome.w3 is None else genome.w3.tolist(),
        "note": "Weights correspond to a tanh MLP: qpos -> 8 -> 8 -> nu; outputs clipped to actuator ctrlrange.",
    }
    try:
        with open(DATA / filename, "w") as f:
            json.dump(ctrl, f, indent=2)
    except Exception as e:
        console.log(f"[WARN] Saving controller info failed: {e!r}")


def plot_results(best_fitness_history: List[float], average_fitness_history: List[float]) -> None:
    """Save and show evolution progress (best vs average)."""
    try:
        plt.figure(figsize=(10, 5))
        gens = np.arange(1, len(best_fitness_history) + 1)
        plt.plot(gens, best_fitness_history, label="Best Fitness")
        plt.plot(gens, average_fitness_history, label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (forward X progress)")
        plt.title("Evolution Progress")
        plt.grid(True)
        plt.legend()
        out = DATA / "evolution_results.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        console.log(f"[plot] Saved evolution plot to {out}")
    except Exception as e:
        console.log(f"[WARN] Could not plot/save evolution results: {e!r}")


# --------------------------- GA loop --------------------------------------
def run_ga(generations: int, pop_size: int, duration: int) -> Tuple[Genome, float, Dict[str, Any], List[float], List[float]]:
    population: List[Genome] = [Genome.random() for _ in range(pop_size)]
    fitness: List[float] = [float("-inf")] * pop_size
    extras:  List[Dict[str, Any]] = [{} for _ in range(pop_size)]

    best_fit = float("-inf")
    best_genome: Genome | None = None
    best_extra: Dict[str, Any] = {}

    best_hist: List[float] = []
    avg_hist:  List[float] = []

    for gen in range(generations):
        console.log(f"=== Generation {gen+1}/{generations} ===")
        # Evaluate
        for i, indiv in enumerate(population):
            fit, ext = evaluate(indiv, duration=duration, mode="simple")
            fitness[i] = fit
            extras[i] = ext

        # Per-gen stats
        gen_best_idx = int(np.argmax(fitness))
        gen_best, gen_best_fit, gen_best_ext = population[gen_best_idx], fitness[gen_best_idx], extras[gen_best_idx]
        gen_avg_fit = float(np.mean(fitness))
        console.log(f"[GEN {gen}] Best fitness = {gen_best_fit:.4f} | Avg = {gen_avg_fit:.4f}")

        # Save best robot JSON for this generation
        try:
            save_graph_as_json(gen_best_ext["robot_graph"], DATA / f"best_robot_gen{gen:03d}.json")
        except Exception as e:
            console.log(f"[WARN] Saving robot graph failed for gen {gen}: {e!r}")

        # Update overall best
        if gen_best_fit > best_fit or best_genome is None:
            best_fit = gen_best_fit
            best_genome = gen_best.copy()
            best_extra = gen_best_ext

        # Save checkpoint for this generation
        save_checkpoint(population, gen_best, gen, fitness, gen_avg_fit, gen_best_fit)

        # Record histories
        best_hist.append(gen_best_fit)
        avg_hist.append(gen_avg_fit)

        # Selection: keep top K
        K = max(2, pop_size // 3)
        parent_idx = list(np.argsort(fitness)[-K:][::-1])
        parents = [population[i] for i in parent_idx]

        # Produce children
        children: List[Genome] = []
        rng = np.random.default_rng()
        while len(children) < pop_size:
            pa, pb = rng.choice(parents, size=2, replace=True)
            c1, c2 = crossover(pa, pb)
            c1 = mutate(c1, sigma=0.05, p=0.25)
            c2 = mutate(c2, sigma=0.10, p=0.30)
            children.append(c1)
            if len(children) < pop_size:
                children.append(c2)

        # Elitism: carry forward the generation's best
        children[0] = gen_best.copy()
        population = children

    assert best_genome is not None
    return best_genome, best_fit, best_extra, best_hist, avg_hist


# --------------------------- MAIN ----------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=GENERATIONS_DEFAULT)
    p.add_argument("--pop", type=int, default=POP_SIZE_DEFAULT)
    p.add_argument("--duration", type=int, default=DURATION_DEFAULT)
    p.add_argument("--mode", type=str, default=RUN_MODE_DEFAULT, choices=["simple","video","launcher","frame","no_control"])
    p.add_argument("--record", action="store_true", help="Also record champion video to videos/champion.mp4")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Run GA (collect histories for plotting)
    best_genome, best_fit, best_extra, best_hist, avg_hist = run_ga(args.generations, args.pop, args.duration)
    console.log(f"[RESULT] Best overall fitness: {best_fit:.4f}")

    # Final champion run in chosen mode + optional video
    video_name = "champion.mp4" if (args.mode == "video" or args.record) else None
    fit, ext = evaluate(best_genome, duration=args.duration, mode=args.mode, video_filename=video_name)

    # Save final robot JSON & controller info
    save_best_robot(ext["robot_graph"], "best_robot_final.json")
    save_final_controller(best_genome, "controller_best_robot_final.json")

    # Save & show evolution plot
    plot_results(best_hist, avg_hist)

    # Path plot at the end (like template)
    show_xpos_history(ext.get("history", []))


if __name__ == "__main__":
    main()
