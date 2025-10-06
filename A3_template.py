"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
from dataclasses import dataclass

# Local libraries
#from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
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

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def show_xpos_history(history: list[float]) -> None:
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
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
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

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()


# Fixed-weights controller factory: returns a callback bound to one individual's weights
def make_nn_controller(w1, w2, w3):
    def ctrl(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        x = data.qpos
        h1 = np.tanh(x @ w1)
        h2 = np.tanh(h1 @ w2)
        u  = np.tanh(h2 @ w3) * (np.pi / 2)  # keep torques/angles sane
        return u
    return ctrl

def build_robot_from_vectors(type_p, conn_p, rot_p):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward([type_p, conn_p, rot_p])
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    graph = hpd.probability_matrices_to_graph(*p_mats)
    core = construct_mjspec_from_graph(graph)
    return core

@dataclass
class Individual:
    # Morphology genes (inputs to NDE)
    type_p: np.ndarray
    conn_p: np.ndarray
    rot_p:  np.ndarray
    # Controller genes (weights)
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    # Bookkeeping
    fitness: float | None = None
    history: list[list[float]] | None = None
    
def evaluate(ind: Individual, duration: int = 15) -> float:
    # clear callbacks (pitfall: must be None at start)
    mj.set_mjcb_control(None)

    # build morphology
    robot = build_robot_from_vectors(ind.type_p, ind.conn_p, ind.rot_p)

    # track the robot "core" geom
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    # fixed-weights controller for this individual
    ctrl = Controller(controller_callback_function=make_nn_controller(ind.w1, ind.w2, ind.w3),
                      tracker=tracker)

    # fast headless evaluation
    experiment(robot=robot, controller=ctrl, duration=duration, mode="simple")

    # store path and compute fitness (forward x progress with fixed horizon)
    path = tracker.history["xpos"][0]
    ind.history = path
    end_x = float(path[-1][0])
    ind.fitness = end_x
    return ind.fitness

# EA hyperparameters
POP = 30
GENS = 25
DURATION = 15
ELITE = 2
TOUR = 3
H = 16                 # hidden width for policy MLP
SIG_BODY = 0.15        # mutation sigma for NDE vectors
SIG_W = 0.20           # mutation sigma for weights
GENOTYPE_SIZE = 64     # your template already uses 64

def baseline_random(pop_size=POP, gens=GENS, duration=DURATION):
    best = None
    bests = []
    for g in range(gens):
        gen_best = None
        for _ in range(pop_size):
            ind = rand_individual()
            in_size, out_size = infer_policy_shapes(ind)
            ind.w1 = RNG.normal(0, 0.3, size=(in_size, H))
            ind.w2 = RNG.normal(0, 0.3, size=(H, H))
            ind.w3 = RNG.normal(0, 0.3, size=(H, out_size))
            evaluate(ind, duration=duration)
            if gen_best is None or ind.fitness > gen_best.fitness:
                gen_best = ind
        bests.append(gen_best)
        if best is None or gen_best.fitness > best.fitness:
            best = gen_best
        print(f"[BASELINE] Gen {g+1:02d} | best x = {gen_best.fitness:.3f}")
    return best, bests


def rand_individual():
    # body genes in [0,1]
    type_p = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    conn_p = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    rot_p  = RNG.random(GENOTYPE_SIZE).astype(np.float32)

    # temporary shapes; we’ll infer true dims and reinit below
    w1 = RNG.normal(0, 0.3, size=(64, H))
    w2 = RNG.normal(0, 0.3, size=(H, H))
    w3 = RNG.normal(0, 0.3, size=(H, 32))
    return Individual(type_p, conn_p, rot_p, w1, w2, w3)

def mutate(ind: Individual) -> Individual:
    child = Individual(
        np.clip(ind.type_p + RNG.normal(0, SIG_BODY, ind.type_p.shape), 0, 1),
        np.clip(ind.conn_p + RNG.normal(0, SIG_BODY, ind.conn_p.shape), 0, 1),
        np.clip(ind.rot_p  + RNG.normal(0, SIG_BODY, ind.rot_p.shape ), 0, 1),
        ind.w1 + RNG.normal(0, SIG_W, ind.w1.shape),
        ind.w2 + RNG.normal(0, SIG_W, ind.w2.shape),
        ind.w3 + RNG.normal(0, SIG_W, ind.w3.shape),
    )
    child.fitness = None
    child.history = None
    return child

def tournament(pop, k=TOUR):
    cand = RNG.choice(pop, size=k, replace=False)
    return max(cand, key=lambda i: -1e9 if i.fitness is None else i.fitness)

def infer_policy_shapes(ind: Individual):
    # Build once to get (input_size, output_size) for MLP shapes based on morphology
    robot = build_robot_from_vectors(ind.type_p, ind.conn_p, ind.rot_p)
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    input_size = len(data.qpos)
    output_size = model.nu
    return input_size, output_size

def evolve():
    # initial population
    pop = [rand_individual() for _ in range(POP)]

    # infer true MLP dims once, then re-init everyone’s weights to match
    in_size, out_size = infer_policy_shapes(pop[0])
    for p in pop:
        p.w1 = RNG.normal(0, 0.3, size=(in_size, H))
        p.w2 = RNG.normal(0, 0.3, size=(H, H))
        p.w3 = RNG.normal(0, 0.3, size=(H, out_size))

    # evaluate initial population
    for ind in pop:
        evaluate(ind, duration=DURATION)

    best_per_gen = []

    for g in range(GENS):
        pop.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
        best_per_gen.append(pop[0])
        next_pop = pop[:ELITE]  # elitism

        # fill population by tournament + mutation
        while len(next_pop) < POP:
            parent = tournament(pop)
            child = mutate(parent)
            evaluate(child, duration=DURATION)
            next_pop.append(child)

        pop = next_pop
        print(f"Gen {g+1:02d} | best x = {best_per_gen[-1].fitness:.3f}")

    pop.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
    return pop[0], best_per_gen


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #


def main() -> None:
    # 1) Train (evolve morphology + controller)
    best, bests = evolve()

    print("\n=== FINAL BEST ===")
    print(f"Best fitness (x): {best.fitness:.3f}")

    # 2) Plot the champion path (lightweight, once)
    show_xpos_history(best.history)

    # 3) Visualize the champion in an animated run (short GUI demo)
    mj.set_mjcb_control(None)
    robot = build_robot_from_vectors(best.type_p, best.conn_p, best.rot_p)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=make_nn_controller(best.w1, best.w2, best.w3),
                      tracker=tracker)
    experiment(robot=robot, controller=ctrl, duration=20, mode="animate")

    # Optional: also run & compare baseline (uncomment to produce baseline logs)
    # base_best, base_bests = baseline_random()




if __name__ == "__main__":
    main()