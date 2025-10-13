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
GENERATIONS = 300          # evolutionary generations
POP_SIZE    = 16           # population size
RUN_MODE    = "launcher"   # "simple" | "video" | "launcher" | "frame" | "no_control"

# ===[ BASELINE & VIDEO SETTINGS ]===========================================
RUN_BASELINE = True          # Run random baseline before GA
BASELINE_TRIALS = 5          # Number of random baseline trials
RECORD_VIDEO_BEST = True     # Enable automatic recording of champion video
# ==========================================================================

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
from typing import TYPE_CHECKING, Any, Literal, cast
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer

# optuna is optional for tuning; import conditionally so static checkers don't error
import importlib.util
optuna_spec = importlib.util.find_spec("optuna")
if optuna_spec is not None:
    import importlib

    optuna = importlib.import_module("optuna")
else:
    optuna = None

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

SCRIPT_NAME = "A3_J"
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

# ---------------------- Controller optimization / novelty -----------------
# If True, for each morphology we run a short controller optimization (ES)
# to find good policy weights before full evaluation.
OPTIMIZE_CONTROLLER = True
CONTROLLER_OPT_DURATION = 3  # seconds for inner optimization evaluations
CONTROLLER_POP = 6           # population per ES iteration
CONTROLLER_ITERS = 3         # ES iterations per morphology
CONTROLLER_SIGMA = 0.25      # initial mutation size for weights

# Novelty search settings
NOVELTY_ENABLED = True
NOVELTY_K = 5
NOVELTY_WEIGHT = 0.5
NOVELTY_ADD_THRESHOLD = 0.5
NOVELTY_ADD_PROB = 0.2


# --------------------------- Template-style helpers -----------------------
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    """Negative Cartesian distance from final position to TARGET_POSITION,
    with small lateral penalty and fall penalty to discourage wandering/falling."""
    if not history:
        return float("-1e9")
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # base: distance to target (minimize)
    d = math.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    forward_score = -float(d)

    # lateral penalty (discourage straying too far from center line y=0)
    lateral_penalty = -0.3 * abs(float(yc))  # small weight so forward progress dominates

    # fall penalty (if very low z, likely fell off or collapsed)
    fall_penalty = -100.0 if float(zc) < 0.05 else 0.0

    return forward_score + lateral_penalty + fall_penalty


def show_xpos_history(history: list[tuple[float, float, float]]) -> None:
    """Template-style path plot over OlympicArena background (no extra plots)."""
    if not history:
        console.log("[show_xpos_history] No history to plot; skipping.")
        return

    # Ensure no control callback interferes and render background using default camera
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    # single_frame_renderer in this repo does not accept a camera kwarg
    single_frame_renderer(model, data, save=True, save_path=save_path)

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
                try:
                    lo[i] = model.actuator_ctrlrange[i, 0]
                    hi[i] = model.actuator_ctrlrange[i, 1]
                except Exception:
                    lo[i] = -1.0
                    hi[i] = 1.0
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
    # optional controller weights (ES/CMA-ES optimized per morphology)
    controller_weights: np.ndarray | None = None

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


def decode_robot(genome: Genome) -> tuple[Any, "DiGraph[Any]"]:
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


# ------------------ Policy weight helpers & simple ES optimizer ----------
def policy_param_count(input_dim: int, hidden: int, out_dim: int) -> int:
    # two-layer MLP: input->hidden (with bias), hidden->out (with bias)
    return input_dim * hidden + hidden + hidden * out_dim + out_dim


def unpack_policy(weights: np.ndarray, input_dim: int, hidden: int, out_dim: int):
    p = weights.astype(np.float32)
    idx = 0
    w1_size = input_dim * hidden
    W1 = p[idx: idx + w1_size].reshape((input_dim, hidden)); idx += w1_size
    b1 = p[idx: idx + hidden]; idx += hidden
    w2_size = hidden * out_dim
    W2 = p[idx: idx + w2_size].reshape((hidden, out_dim)); idx += w2_size
    b2 = p[idx: idx + out_dim]; idx += out_dim
    return W1, b1, W2, b2


def policy_forward(weights: np.ndarray, obs: np.ndarray, hidden: int, out_dim: int) -> np.ndarray:
    input_dim = obs.shape[0]
    expected = input_dim * hidden + hidden + hidden * out_dim + out_dim
    # Adjust if weights are too short or too long
    if weights.size < expected:
        weights = np.pad(weights, (0, expected - weights.size))
    elif weights.size > expected:
        weights = weights[:expected]

    W1, b1, W2, b2 = unpack_policy(weights, input_dim, hidden, out_dim)
    h = np.tanh(obs @ W1 + b1)
    out = np.tanh(h @ W2 + b2)
    return out


class NoveltyArchive:
    def __init__(self):
        self.archive: list[np.ndarray] = []

    def descriptor(self, history: list[tuple[float, float, float]]) -> np.ndarray:
        if not history:
            return np.zeros(3, dtype=float)
        arr = np.array(history, dtype=float)
        x = arr[:, 0]
        return np.array([x[-1], float(np.mean(x)), float(np.std(x))], dtype=float)

    def novelty(self, desc: np.ndarray, k: int = 5) -> float:
        if not self.archive:
            return float('inf')
        dists = [float(np.linalg.norm(desc - a)) for a in self.archive]
        dists.sort()
        k = min(k, len(dists))
        return float(np.mean(dists[:k]))

    def maybe_add(self, desc: np.ndarray, threshold: float = NOVELTY_ADD_THRESHOLD) -> bool:
        n = self.novelty(desc, NOVELTY_K)
        if n >= threshold or (np.random.random() < NOVELTY_ADD_PROB):
            self.archive.append(desc.copy())
            return True
        return False


def optimize_controller_weights(genome: Genome, input_dim: int, hidden: int, out_dim: int) -> np.ndarray:
    param_count = policy_param_count(input_dim, hidden, out_dim)
    rng = np.random.default_rng()
    mu = rng.normal(0.0, 0.05, size=(param_count,)).astype(np.float32)
    sigma = CONTROLLER_SIGMA

    for _ in range(CONTROLLER_ITERS):
        pop = [mu + rng.normal(0.0, sigma, size=mu.shape).astype(np.float32) for _ in range(CONTROLLER_POP)]
        scores = []
        for w in pop:
            genome.controller_weights = w
            fit, _ = evaluate(genome, duration=CONTROLLER_OPT_DURATION, mode="simple")
            scores.append(fit)

        best_idx = int(np.argmax(scores))
        mu = 0.7 * mu + 0.3 * pop[best_idx]
        sigma *= 0.95

    return mu


# --------------------------- EVALUATION / SIM --------------------------------
def evaluate(genome: Genome, duration: int, mode: ViewerTypes = "simple",
             video_filename: str | None = None) -> tuple[float, dict[str, Any]]:
    """Build, run, and score one individual; optionally record a video."""
    # Clear previous callbacks (common pitfall)
    mj.set_mjcb_control(None)

    # World & robot
    world = OlympicArena()
    core, robot_graph = decode_robot(genome)

    # --- EARLY STOP: limb count heuristic (*** ADDED) ---
    # count hinge-like modules; if too few, abort early with terrible fitness
    try:
        hinge_count = 0
        for _, data in robot_graph.nodes(data=True):
            # node data may store a module type under different keys; be permissive
            found = False
            for key in ("module_type", "type", "module"):
                if key in data and isinstance(data[key], str):
                    if "hinge" in data[key].lower():
                        found = True
                        break
            # fallback: check any stringified values
            if not found:
                for v in data.values():
                    if isinstance(v, str) and "hinge" in v.lower():
                        found = True
                        break
            if found:
                hinge_count += 1
    except Exception:
        hinge_count = 0

    if hinge_count < 3:
        # too few limbs, skip expensive sim
        return float("-1e9"), {"robot_graph": robot_graph, "reason": "few_limbs"}

    # Adaptive spawn height (avoid clipping with larger bodies)
    approx_size = 0.02 * GENOTYPE_SIZE  # heuristic
    spawn_z = max(SPAWN_POS[2], 0.25 + 0.001 * approx_size)
    # BaseWorld.spawn uses 'position' kwarg
    world.spawn(core.spec, position=[SPAWN_POS[0], SPAWN_POS[1], spawn_z])



    # Model/data
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state
    mj.mj_resetData(model, data)

    # Controller: either use controller_weights (policy) or seeded MLP
    if getattr(genome, "controller_weights", None) is not None:
        # build a lightweight policy-based controller
        weights = genome.controller_weights

        def policy_callback(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
            # observation: qpos concat qvel (flattened)
            obs = np.concatenate([data.qpos, data.qvel]).astype(np.float32)
            out = policy_forward(weights, obs, hidden=32, out_dim=model.nu)
            ctrl = np.asarray(out * genome.ctrl_scale, dtype=np.float64)
            # clip by actuator ranges where possible
            nu = model.nu
            if nu > 0:
                lo = np.empty(nu); hi = np.empty(nu)
                for i in range(nu):
                    try:
                        lo[i] = model.actuator_ctrlrange[i,0]
                        hi[i] = model.actuator_ctrlrange[i,1]
                    except Exception:
                        lo[i] = -1.0; hi[i] = 1.0
                ctrl = np.clip(ctrl, lo, hi)
            return ctrl

        ctrl = Controller(controller_callback_function=policy_callback, tracker=Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core"))
    else:
        seed_uint32 = int(genome.ctrl_u * (2**32 - 1)) & 0xFFFFFFFF
        ctrl = make_nn_controller(seed_uint32, genome.ctrl_scale)

    # Tracker exists by default; set it up
    ctrl.tracker.setup(world.spec, data)

    # Register typed MuJoCo callback
    def _mjcb_control(m: mj.MjModel, d: mj.MjData) -> None:
        ctrl.set_control(m, d)

    mj.set_mjcb_control(_mjcb_control)

    # Run simulation
    if mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "video":
        assert video_filename is not None, "video_filename required for mode='video'"
        video_dir = DATA / "videos"
        video_dir.mkdir(exist_ok=True, parents=True)
        # VideoRecorder uses file_name & output_folder in this repo
        video_recorder = VideoRecorder(file_name=video_filename, output_folder=str(video_dir))
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
    history: list[tuple[float, float, float]] = []
    # Tracker exists; check for collected xpos history
    if "xpos" in ctrl.tracker.history and len(ctrl.tracker.history["xpos"]) > 0:
        history = ctrl.tracker.history["xpos"][0]
    fit = fitness_function(history)

    return fit, {"history": history, "robot_graph": robot_graph}


# --------- Non-learner short test & dynamic duration schedule -------------
def quick_displacement_test(genome: Genome) -> float:
    """Run a very short simulation and return XY displacement in meters."""
    _, extra = evaluate(genome, duration=NONLEARNER_TEST_DUR, mode="simple")
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
def blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    lam = np.random.default_rng().uniform(-alpha, 1 + alpha, size=a.shape).astype(np.float32)
    c1 = lam * a + (1 - lam) * b
    c2 = lam * b + (1 - lam) * a
    return c1, c2

def crossover(pa: Genome, pb: Genome) -> tuple[Genome, Genome]:
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


# ------------------ Random baseline (adapted from template) ---------------
def baseline_random(pop_size: int = 30, gens: int = 25, duration: int = DUR_START) -> tuple[Genome, list[Genome]]:
    """Simple random-baseline: random genomes, no GA, keep best per generation."""
    best: Genome | None = None
    bests: list[Genome] = []
    for g in range(gens):
        gen_best = None
        for _ in range(pop_size):
            ind = Genome.random()
            # do NOT apply ensure_learner here to remain a true random baseline
            fit, extra = evaluate(ind, duration=duration, mode="simple")
            ind_fit = fit
            if gen_best is None or ind_fit > getattr(gen_best, "fitness", float("-inf")):
                # store fitness into object for convenience
                ind_copy = ind.copy()
                setattr(ind_copy, "fitness", ind_fit)
                setattr(ind_copy, "history", extra.get("history", []))
                gen_best = ind_copy
        assert gen_best is not None
        bests.append(gen_best)
        if best is None or gen_best.fitness > getattr(best, "fitness", float("-inf")):
            best = gen_best
        console.log(f"[BASELINE] Gen {g+1:02d} | best fitness = {gen_best.fitness:.3f}")
    # If best remained None (all gens failed), return a dummy genome alongside bests
    if best is None:
        dummy = Genome.random()
        return dummy, bests
    return best, bests

# ===[ Enhanced random baseline runner ]====================================
import json

def run_random_baseline(trials: int, duration: int, data_dir: Path) -> dict:
    """Run a few random robots (no evolution) to compare GA vs random."""
    console.log(f"[BASELINE] Running {trials} random trials...")
    results = []
    for i in range(trials):
        genome = Genome.random()
        fit, _ = evaluate(genome, duration=duration, mode="simple")
        results.append(fit)
        console.log(f"[BASELINE] Trial {i+1}/{trials}: {fit:.4f}")

    avg = float(np.mean(results))
    best = float(np.max(results))
    data = {"average": avg, "best": best, "all": results}

    with open(data_dir / "baseline_results.json", "w") as f:
        json.dump(data, f, indent=2)
    console.log(f"[BASELINE] Done → avg={avg:.4f}, best={best:.4f}")
    return data
# ==========================================================================

# --- GA runner (now logs + accepts mutate params) ---
def run_ga(generations: int, pop_size: int, mutate_sigma: float = 0.10, mutate_p: float = 0.3) -> tuple[Genome, float, dict[str, Any]]:
    # Init population with non-learner filtering
    population: list[Genome] = [ensure_learner(Genome.random()) for _ in range(pop_size)]
    fitness: list[float] = [float("-inf")] * pop_size
    extras:  list[dict[str, Any]] = [{} for _ in range(pop_size)]

    best_fit = float("-inf")
    best_genome: Genome | None = None
    best_extra: dict[str, Any] = {}

    # logging arrays (*** ADDED)
    best_fitnesses: list[float] = []
    mean_fitnesses: list[float] = []
    std_fitnesses: list[float] = []

    # novelty archive
    archive = NoveltyArchive() if NOVELTY_ENABLED else None

    for gen in range(generations):
        # Dynamic duration based on progress so far
        dur = schedule_duration(best_fit)
        console.log(f"=== Generation {gen+1}/{generations} (dur={dur}s) ===")

        # Evaluate population
        for i, indiv in enumerate(population):
            fit, ext = evaluate(indiv, duration=dur, mode="simple")
            fitness[i] = fit
            extras[i]  = ext

        # collect stats (*** ADDED)
        # fitness is list[float]; build array directly
        arr = np.array(fitness, dtype=float)
        mean_f = float(np.mean(arr))
        std_f = float(np.std(arr))
        gen_best_value = float(np.max(arr))
        best_fitnesses.append(gen_best_value)
        mean_fitnesses.append(mean_f)
        std_fitnesses.append(std_f)

        # Pick best
        idx = int(np.argmax(fitness))
        gen_best, gen_best_fit, gen_best_ext = population[idx], fitness[idx], extras[idx]
        console.log(f"[GEN {gen}] Best fitness = {gen_best_fit:.4f}")

        # record novelty for gen best and maybe add to archive
        if archive is not None:
            desc = archive.descriptor(gen_best_ext.get("history", []))
            nscore = archive.novelty(desc, NOVELTY_K)
            console.log(f"[GEN {gen}] Novelty = {nscore:.3f}")
            archive.maybe_add(desc)

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

        # Selection: use tournament selection for parent sampling
        def tournament_select(pop: list[Genome], fit_arr: list[float], k: int = 3) -> Genome:
            # sample k individuals and return the best among them
            idxs = np.random.default_rng().choice(len(pop), size=k, replace=False)
            # use blended score fitness + novelty to encourage diverse parents
            best_i = idxs[0]
            best_v = fit_arr[best_i]
            # compute novelty for each candidate if archive exists
            for ii in idxs[1:]:
                score = fit_arr[ii]
                if archive is not None:
                    hist = extras[ii].get("history", [])
                    desc = archive.descriptor(hist)
                    n = archive.novelty(desc, NOVELTY_K)
                    score = score + NOVELTY_WEIGHT * n
                if score > best_v:
                    best_i = ii
                    best_v = score
            return pop[best_i]

        # Offspring via tournament selection
        children: list[Genome] = []
        while len(children) < pop_size:
            pa = tournament_select(population, fitness, k=3)
            pb = tournament_select(population, fitness, k=3)
            c1, c2 = crossover(pa, pb)
            c1 = ensure_learner(mutate(c1, sigma=mutate_sigma, p=mutate_p))
            c2 = ensure_learner(mutate(c2, sigma=mutate_sigma, p=mutate_p))

            # optional controller optimization per morphology
            if OPTIMIZE_CONTROLLER:
                # quick input_dim estimate: qpos+qvel (use model size heuristic)
                try:
                    # decode robot to get spec and model size
                    core1, _ = decode_robot(c1)
                    # compile model for dimension estimates
                    tmp_world = OlympicArena()
                    tmp_world.spawn(core1.spec, position=[SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2]])
                    m = tmp_world.spec.compile()
                    input_dim = len(m.nq) + len(m.nv) if hasattr(m, 'nq') and hasattr(m, 'nv') else 32
                    out_dim = m.nu if hasattr(m, 'nu') else 8
                except Exception:
                    input_dim = 32
                    out_dim = 8
                hid = 32
                c1.controller_weights = optimize_controller_weights(c1, input_dim, hid, out_dim)
                # repeat for c2 if room
                if len(children) + 1 < pop_size:
                    try:
                        core2, _ = decode_robot(c2)
                        tmp_world2 = OlympicArena()
                        tmp_world2.spawn(core2.spec, position=[SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2]])
                        m2 = tmp_world2.spec.compile()
                        input_dim2 = len(m2.nq) + len(m2.nv) if hasattr(m2, 'nq') and hasattr(m2, 'nv') else input_dim
                        out_dim2 = m2.nu if hasattr(m2, 'nu') else out_dim
                    except Exception:
                        input_dim2, out_dim2 = input_dim, out_dim
                    c2.controller_weights = optimize_controller_weights(c2, input_dim2, hid, out_dim2)
            children.append(c1)
            if len(children) < pop_size:
                children.append(c2)

        # Elitism
        children[0] = gen_best.copy()
        population = children

    # Save fitness logs (*** ADDED)
    try:
        log_path = DATA / f"fitness_logs_pop{pop_size}_gen{generations}.npz"
        np.savez_compressed(log_path, best=np.array(best_fitnesses), mean=np.array(mean_fitnesses), std=np.array(std_fitnesses))
        console.log(f"[LOG] Saved fitness logs: {log_path}")
    except Exception as e:
        console.log(f"[WARN] Failed to save fitness logs: {e!r}")

    assert best_genome is not None
    return best_genome, best_fit, best_extra, best_fitnesses, mean_fitnesses


def plot_results(best_fitness_history: list[float],
                 average_fitness_history: list[float],
                 data_dir: Path,
                 baseline_data: dict | None = None) -> None:
    """Save and show evolution progress (best vs average + optional baseline)."""
    try:
        plt.figure(figsize=(10, 5))
        gens = np.arange(1, len(best_fitness_history) + 1)
        plt.plot(gens, best_fitness_history, label="Best Fitness")
        plt.plot(gens, average_fitness_history, label="Average Fitness")

        if baseline_data:
            plt.axhline(y=baseline_data["average"], color="r", linestyle="--", label="Random Baseline (avg)")
            plt.axhline(y=baseline_data["best"], color="r", linestyle=":", label="Random Baseline (best)")

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Evolution Progress vs Random Baseline")
        plt.grid(True)
        plt.legend()
        out = data_dir / "evolution_results.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        console.log(f"[plot] Saved evolution plot to {out}")
    except Exception as e:
        console.log(f"[WARN] Could not plot/save evolution results: {e!r}")


# --------------------------- OPTUNA TUNING (outer GA) ----------------------
def tune_ga(n_trials: int = 10, eval_sim_seconds: int = 2) -> dict[str, Any]:
    """Tune basic GA hyperparams with Optuna. Does NOT run full budget."""
    def objective(trial: Any) -> float:
        pop = trial.suggest_int("pop_size", 16, 48, step=8)
        gens = trial.suggest_int("generations", 10, 60, step=10)
        sigma = trial.suggest_float("mutate_sigma", 0.02, 0.4, log=True)
        p = trial.suggest_float("mutate_p", 0.05, 0.45)

        # short-run GA for tuning
        # note: ensure_learner and decode will still run, so tuning costs are nontrivial
        _, bf, _ = run_ga(gens, pop, mutate_sigma=sigma, mutate_p=p)
        # we want to minimize (optuna) -> return negative best fitness
        return -bf

    if optuna is None:
        console.log("[OPTUNA] optuna not available; skipping tuning")
        return {}
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    console.log(f"[OPTUNA] Best params: {study.best_params}")
    return cast(dict[str, Any], study.best_params)


# --------------------------- MAIN ----------------------------------------
def main() -> None:
    # Optional baseline before GA
    baseline_data = run_random_baseline(BASELINE_TRIALS, DUR_START, DATA) if RUN_BASELINE else None

    # Run GA and get fitness histories
    best_genome, best_fit, best_extra, best_hist, avg_hist = run_ga(GENERATIONS, POP_SIZE)
    console.log(f"[RESULT] Best overall fitness: {best_fit:.4f}")

    # Plot GA vs baseline
    plot_results(best_hist, avg_hist, DATA, baseline_data)

    # Final champion evaluation (with video if enabled)
    video_name = "champion.mp4" if RECORD_VIDEO_BEST else None
    fit, ext = evaluate(best_genome, duration=DUR_MAX, mode="video" if RECORD_VIDEO_BEST else RUN_MODE, video_filename=video_name)

    # Save robot and genome
    try:
        save_graph_as_json(ext["robot_graph"], DATA / "best_robot_final.json")
    except Exception as e:
        console.log(f"[WARN] Saving final robot graph failed: {e!r}")

    np.savez_compressed(
        DATA / "best_genome.npz",
        type_p=best_genome.type_p,
        conn_p=best_genome.conn_p,
        rot_p=best_genome.rot_p,
        ctrl_u=np.array([best_genome.ctrl_u]),
        ctrl_scale=np.array([best_genome.ctrl_scale]),
    )
    console.log("[SAVE] Best genome and graph saved.")

    # Plot path
    show_xpos_history(ext.get("history", []))


if __name__ == "__main__":
    main()