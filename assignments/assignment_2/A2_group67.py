
# Standard library
from pathlib import Path
from typing import Literal
import csv

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.ec import Population, config
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
#from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
#set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options

FALL_HEIGHT: float = 0.05


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    return gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #


# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6
POP_SIZE: int = 80


def nn_controller(model: mj.MjModel, data: mj.MjData, weights: list[npt.NDArray[np.float64]],) -> npt.NDArray[np.float64]:

    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    inputs = data.qpos

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:

    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]

def flatten_weights(weights: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Flatten [w1, w2] into one 1D genotype vector."""
    return np.concatenate([w.ravel() for w in weights])

def unflatten_weights(flat: npt.NDArray[np.float64], input_size: int, output_size: int):
    """Reshape a flat genotype back into [w1, w2] for nn_controller."""
    split = input_size * HIDDEN_SIZE
    w1 = flat[:split].reshape(input_size, HIDDEN_SIZE)
    w2 = flat[split:].reshape(HIDDEN_SIZE, output_size)
    return [w1, w2]


# ============================================================================ #
#  3. Individuals and Population
# ============================================================================ #


def make_individual(
    input_size: int,
    output_size: int,
) -> npt.NDArray[np.float64]:
    """Create one random individual: a flattened genotype (weight vector)."""
    weights = make_random_weights(input_size, output_size)
    return flatten_weights(weights)


def init_population(pop_size: int, input_size: int, output_size: int,
) -> list[npt.NDArray[np.float64]]:
    return [make_individual(input_size, output_size) for _ in range(pop_size)]

# ============================================================================ #
#  Survivor selection
# ============================================================================ #

def survivor_selection(population: Population) -> Population:

    candidates = list(population.alive)
    target_size = config.target_population_size

    if not 1 <= target_size <= len(candidates):
        raise ValueError("Invalid target population size.")

    for ind in candidates:
        if ind.requires_eval or ind.fitness_ is None:
            raise ValueError("All candidates must have evaluated fitness.")
    
    # Sort candidates by fitness (lower is better)
    sorted_candidates = sorted(candidates, key=lambda ind: float(ind.fitness_))

    # Marking the indiviuals that are not selected for the next generation as dead
    for ind in sorted_candidates[target_size:]:
        ind.alive = False

    return population



# ============================================================================ #
#  4. POSITION AND FITNESS
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def get_min_z_height(data: mj.MjData) -> float:
    """Return the minimum z height of the robot core during the simulation."""
    return float(np.min(data.qpos[2]))

def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
    min_z_height: float,
):

    if min_z_height < FALL_HEIGHT:
        return 10.0

    target = np.asarray(TARGET_POSITION)
    initial_dist = float(np.linalg.norm(initial_position[:2] - target[:2]))
    final_dist = float(np.linalg.norm(final_position[:2] - target[:2]))
    return final_dist - initial_dist


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(
    mode: ViewerTypes = MODE,
    individual: npt.NDArray[np.float64] | None = None,
) -> float:
    """Pass `individual` to replay a specific genotype; omit it for random weights."""
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(data.qpos)
    output_size = model.nu

    weights = (
        unflatten_weights(individual, input_size, output_size)
        if individual is not None
        else make_random_weights(input_size, output_size)
    )

    # Initialize minimum z height of the robot.
    min_z_height = get_min_z_height(data)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        nonlocal min_z_height
        min_z_height = min(min_z_height, get_min_z_height(d))

        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position, min_z_height)

    console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    config.target_population_size = POP_SIZE
    population = init_population(POP_SIZE, input_size, output_size)
    console.log(f"population size                     : {len(population)}")

    fitnesses = [run_experiment(mode="simple", individual=ind) for ind in population]
    console.log(f"best fitness  : {min(fitnesses):.4f}")
    console.log(f"mean fitness  : {np.mean(fitnesses):.4f}")
    console.log(f"worst fitness : {max(fitnesses):.4f}")

    results_path = DATA / "fitness_results.csv"
    with results_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["individual", "fitness"])
        writer.writerows(enumerate(fitnesses))

    # Watch the best individual found move in the viewer.
    best_individual = population[int(np.argmin(fitnesses))]
    run_experiment(mode=MODE, individual=best_individual)


if __name__ == "__main__":
    main()


