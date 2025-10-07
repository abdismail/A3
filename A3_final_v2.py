
"""
A3_final_v2.py — Assignment 3 (Robot Olympics) — Template-faithful + Improvements
-------------------------------------------------------------------------------
- Keeps the original A3 controller style (3-layer tanh MLP).
- Evolves BOTH morphology (NDE input vectors) and controller seed/scale.
- Adds two improvements requested:
  1) Kill-off non-learners via a short displacement test.
  2) Dynamic simulation duration that grows toward a 120s cap as fitness improves.

No extra plots beyond the template path plot.
"""

# ----------------------------- USER EDITABLES -----------------------------
GENERATIONS = 250          # evolutionary generations
POP_SIZE    = 32          # population size
RUN_MODE    = "launcher"  # "simple" | "video" | "launcher" | "frame" | "no_control"
RECORD_VIDEO_BEST = False # also record the final champion if True

# Dynamic duration schedule (seconds)
DUR_START   = 15          # early gens / far from goal
DUR_MID     = 45          # mid progress
DUR_LATE    = 100         # close to goal
DUR_MAX     = 120         # cap; aim is to finish under 120s

# Non-learner filter (short shake test)
NONLEARNER_TEST_DUR  = 3       # seconds
NONLEARNER_MIN_DISP  = 0.20    # meters; minimum displacement to keep
NONLEARNER_MAX_TRIES = 12      # retries before giving up

# ----------------------------- Imports -----------------------------------
from typing import TYPE_CHECKING, Any, Literal, List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import math

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

SCRIPT_NAME = "A3_final_v2"
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# Arena and target
SPAWN_POS = [-0.8, 0.0, 0.35]        # slightly higher to avoid ground clipping
TARGET_POSITION = [5.0, 0.0, 0.5]

# NDE/morphology parameters
NUM_OF_MODULES = 30
GENOTYPE_SIZE  = 64

# Controller scale range (keep safe to avoid QACC explosions)
CTRL_SCALE_RANGE = (0.15 * math.pi, 1.25 * math.pi)

# --------------------------- Template-style helpers -----------------------
def fitness_function(history: List[Tuple[float, float, float]]) -> float:
    """Negative Cartesian distance from final position to TARGET_POSITION."""
    if not history:
        return float("-1e9")
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    d = math.sqrt((xt - xc)**2 + (yt - yc)**2 + (zt - zc)**2)
    return -float(d)


def show_xpos_history(history: List[Tuple[float, float, float]]) -> None:
    """Template-style path plot over OlympicArena background (no extra plots)."""
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

    # Calculate initial position (pixel anchors measured for provided background)
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


# --------------------------- Controller (template-style) ------------------
def make_nn_controller(controller_seed_uint32: int, ctrl_scale: float) -> Controller:
    """Same style as A3: 3-layer tanh MLP; seeded weights; global scale."""
    seed_rng = np.random.default_rng(controller_seed_uint32)

    def callback(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        input_size = len(data.qpos)
        hidden_size = 8
        output_size = model.nu

        # deterministic weights
        w1 = seed_rng.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
        w2 = seed_rng.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
        w3 = seed_rng.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

        inputs = data.qpos
        layer1 = np.tanh(np.dot(inputs, w1))
        layer2 = np.tanh(np.dot(layer1, w2))
        outputs = np.tanh(np.dot(layer2, w3))

        ctrl = outputs * ctrl_scale

        nu = model.nu
        if nu > 0:
            lo = np.empty(nu)
            hi = np.empty(nu)
            for i in range(nu):
                lo[i] = model.actuator_ctrlrange[i, 0] if model.actuator_ctrlrange is not None else -1.0
                hi[i] = model.actuator_ctrlrange[i, 1] if model.actuator_ctrlrange is not None else 1.0
            ctrl = np.clip(ctrl, lo, hi)

        return ctrl.astype(np.float64)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    return Controller(controller_callback_function=callback, tracker=tracker)


# --------------------------- GA setup -------------------------------------
@dataclass
class Genome:
    # Morphology (NDE input vectors)
    type_p: np.ndarray  # (GENOTYPE_SIZE,) in [0,1]
    conn_p: np.ndarray  # (GENOTYPE_SIZE,) in [0,1]
    rot_p:  np.ndarray  # (GENOTYPE_SIZE,) in [0,1]
    # Controller (seed + scale)
    ctrl_u: float       # in [0,1) -> mapped to uint32 seed
    ctrl_scale: float   # in CTRL_SCALE_RANGE

    @staticmethod
    def random() -> "Genome":
        r = np.random.default_rng()
        return Genome(
            type_p=r.random(GENOTYPE_SIZE, dtype=np.float32),
            conn_p=r.random(GENOTYPE_SIZE, dtype=np.float32),
            rot_p= r.random(GENOTYPE_SIZE, dtype=np.float32),
            ctrl_u=float(r.random()),  # [0,1)
            ctrl_scale=float(r.uniform(*CTRL_SCALE_RANGE)),
        )

    def copy(self) -> "Genome":
        return Genome(
            self.type_p.copy(),
            self.conn_p.copy(),
            self.rot_p.copy(),
            float(self.ctrl_u),
            float(self.ctrl_scale),
        )


def decode_robot(genome: Genome) -> Tuple[Any, "DiGraph[Any]"]:
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward([genome.type_p, genome.conn_p, genome.rot_p])

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: "DiGraph[Any]" = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    core = construct_mjspec_from_graph(robot_graph)
    return core, robot_graph


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

    # Controller from deterministic seed & scale
    seed_uint32 = int(genome.ctrl_u * (2**32 - 1)) & 0xFFFFFFFF
    ctrl = make_nn_controller(seed_uint32, genome.ctrl_scale)
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


# --------- Non-learner short test & dynamic duration schedule -------------
def quick_displacement_test(genome: Genome) -> float:
    """Run a very short simulation and return XY displacement in meters."""
    fit, extra = evaluate(genome, duration=NONLEARNER_TEST_DUR, mode="simple")
    hist = extra.get("history", [])
    if not hist:
        return 0.0
    x0, y0, _ = hist[0]
    x1, y1, _ = hist[-1]
    return float(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))

def ensure_learner(genome: Genome) -> Genome:
    """Retry random bodies until a minimal displacement is achieved or tries exhausted."""
    for _ in range(NONLEARNER_MAX_TRIES):
        if quick_displacement_test(genome) >= NONLEARNER_MIN_DISP:
            return genome
        genome = Genome.random()
    return genome  # keep last attempt

def schedule_duration(best_fit_so_far: float) -> int:
    """Return a duration based on progress (fitness is negative distance)."""
    if best_fit_so_far in (float("-inf"), float("-1e9")):
        return DUR_START
    d = abs(best_fit_so_far)  # distance to target (m)
    if d > 4.0:
        return DUR_START
    if d > 2.5:
        return DUR_MID
    if d > 1.2:
        return DUR_LATE
    return DUR_MAX


# --- GA operators (blend crossover + gaussian mutation + elitism) ---
def blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.random.default_rng().uniform(-alpha, 1 + alpha, size=a.shape).astype(np.float32)
    c1 = lam * a + (1 - lam) * b
    c2 = lam * b + (1 - lam) * a
    return c1, c2

def crossover(pa: Genome, pb: Genome) -> Tuple[Genome, Genome]:
    c1, c2 = pa.copy(), pb.copy()
    c1.type_p, c2.type_p = blend(pa.type_p, pb.type_p)
    c1.conn_p, c2.conn_p = blend(pa.conn_p, pb.conn_p)
    c1.rot_p,  c2.rot_p  = blend(pa.rot_p,  pb.rot_p)
    # controller genes: arithmetic blend
    c1.ctrl_u     = float(np.clip(0.5 * (pa.ctrl_u + pb.ctrl_u), 0.0, 1.0))
    c1.ctrl_scale = float(np.clip(0.5 * (pa.ctrl_scale + pb.ctrl_scale), *CTRL_SCALE_RANGE))
    c2.ctrl_u     = c1.ctrl_u
    c2.ctrl_scale = c1.ctrl_scale
    return c1, c2

def mutate(g: Genome, sigma: float = 0.10, p: float = 0.3) -> Genome:
    out = g.copy()
    rng = np.random.default_rng()
    # mutate morphology
    for arr in [out.type_p, out.conn_p, out.rot_p]:
        mask = rng.random(arr.shape) < p
        arr[mask] += rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
        np.clip(arr, 0.0, 1.0, out=arr)
    # mutate controller
    if rng.random() < 0.5:
        out.ctrl_u = float((out.ctrl_u + float(rng.normal(0.0, 0.05))) % 1.0)
    if rng.random() < 0.5:
        out.ctrl_scale = float(np.clip(out.ctrl_scale + float(rng.normal(0.0, 0.15*math.pi)), *CTRL_SCALE_RANGE))
    return out


def run_ga(generations: int, pop_size: int) -> Tuple[Genome, float, Dict[str, Any]]:
    # Init population with non-learner filtering
    population: List[Genome] = [ensure_learner(Genome.random()) for _ in range(pop_size)]
    fitness: List[float] = [float("-inf")] * pop_size
    extras:  List[Dict[str, Any]] = [{} for _ in range(pop_size)]

    best_fit = float("-inf")
    best_genome: Genome | None = None
    best_extra: Dict[str, Any] = {}

    for gen in range(generations):
        # Dynamic duration based on progress so far
        dur = schedule_duration(best_fit)
        console.log(f"=== Generation {gen+1}/{generations} (dur={dur}s) ===")

        # Evaluate population
        for i, indiv in enumerate(population):
            fit, ext = evaluate(indiv, duration=dur, mode="simple")
            fitness[i] = fit
            extras[i]  = ext

        # Pick best
        idx = int(np.argmax(fitness))
        gen_best, gen_best_fit, gen_best_ext = population[idx], fitness[idx], extras[idx]
        console.log(f"[GEN {gen}] Best fitness = {gen_best_fit:.4f}")

        # Save per-gen best body JSON
        try:
            save_graph_as_json(gen_best_ext["robot_graph"], DATA / f"best_robot_gen{gen:03d}.json")
        except Exception as e:
            console.log(f"[WARN] Saving robot graph failed for gen {gen}: {e!r}")

        # Update global best
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_genome = gen_best.copy()
            best_extra  = gen_best_ext

        # Selection: keep top K
        K = max(2, pop_size // 4)
        parent_idx = list(np.argsort(fitness)[-K:][::-1])
        parents = [population[i] for i in parent_idx]

        # Offspring
        children: List[Genome] = []
        while len(children) < pop_size:
            pa, pb = np.random.default_rng().choice(parents, size=2, replace=True)
            c1, c2 = crossover(pa, pb)
            c1 = ensure_learner(mutate(c1, sigma=0.10, p=0.3))
            c2 = ensure_learner(mutate(c2, sigma=0.10, p=0.3))
            children.append(c1)
            if len(children) < pop_size:
                children.append(c2)

        # Elitism
        children[0] = gen_best.copy()
        population = children

    assert best_genome is not None
    return best_genome, best_fit, best_extra


# --------------------------- MAIN ----------------------------------------
def main() -> None:
    best_genome, best_fit, best_extra = run_ga(GENERATIONS, POP_SIZE)
    console.log(f"[RESULT] Best overall fitness: {best_fit:.4f}")

    # Final champion run: use the MAX duration goal (120s) to try to finish arena
    video_name = "champion.mp4" if (RUN_MODE == "video" or RECORD_VIDEO_BEST) else None
    fit, ext = evaluate(best_genome, duration=DUR_MAX, mode=RUN_MODE, video_filename=video_name)

    # Save final robot JSON
    try:
        save_graph_as_json(ext["robot_graph"], DATA / "best_robot_final.json")
    except Exception as e:
        console.log(f"[WARN] Saving final robot graph failed: {e!r}")

    # Original-template path plot at the end
    show_xpos_history(ext.get("history", []))


if __name__ == "__main__":
    main()
