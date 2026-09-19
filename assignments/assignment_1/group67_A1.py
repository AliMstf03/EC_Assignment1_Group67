# Assignment 1 - Group 67
# import the functions that are already maade in the example 
from ariel.ec import EA, EAOperation, Individual, Population, config


# Standard library
import random
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "tree"  # "nde" | "tree" 
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]


# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #


def load_targets(target_dir: Path = TARGET_DIR):
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


# ============================================================================ #
#  2. THE GENOTYPE CONTRACT
# ============================================================================ #

GENOTYPE_SIZE: int = 64  # length of each of the three NDE gene vectors

def make_individual() -> Individual:
    individual = Individual()
    individual.genotype = random_tree(NUM_OF_MODULES)
    return individual

# ============================================================================ #
#  3. THE PARENT SELECTION
# ============================================================================ #


def parent_selection(population: Population) -> Population:

    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ <= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    return shuffled


# ============================================================================ #
#  4. CROSSOVER
# ============================================================================ #

def crossover(population: Population) -> Population:
    # Pak alleen de individuen die als ouder geselecteerd zijn
    parents = population.where(
        lambda ind: bool(ind.tags.get("selected", False))
    )

    # Pak telkens twee ouders
    for idx in range(0, len(parents) - 1, 2):
        parent_a = parents[idx]
        parent_b = parents[idx + 1]

        # Wissel twee willekeurige takken uit
        genome_a, genome_b = crossover_subtree(
            parent_a.genotype,
            parent_b.genotype,
        )

        # Maak het eerste kind
        child_a = Individual()
        child_a.genotype = genome_a
        child_a.tags = {"mutate": True}

        # Maak het tweede kind
        child_b = Individual()
        child_b.genotype = genome_b
        child_b.tags = {"mutate": True}

        # Voeg de kinderen toe aan de populatie
        population.extend([child_a, child_b])

    return population


# ============================================================================ #
#  5. MUTATION
# ============================================================================ #


def mutation_point(population: Population) -> Population:
    for child in population:
        # Alleen de kinderen die net door crossover gemaakt zijn
        if not child.tags.get("mutate", False):
            continue

        # Tag eraf halen, anders wordt het kind de volgende generatie
        # nog een keer gemuteerd
        child.tags = {"mutate": False}

        # Verander 1 module van het kind
        mutate_replace_node(child.genotype)

        # Genoom is veranderd, dus fitness moet opnieuw berekend worden
        child.requires_eval = True

    return population


def mutation_subtree(population: Population) -> Population:
    for child in population:
        # Alleen de kinderen die net door crossover gemaakt zijn
        if not child.tags.get("mutate", False):
            continue

        # Tag eraf halen, anders wordt het kind de volgende generatie
        # nog een keer gemuteerd
        child.tags = {"mutate": False}

        # Vervang een tak van het kind door een nieuwe random tak
        mutate_subtree_replacement(
            child.genotype,
            max_modules=NUM_OF_MODULES,
        )

        # Genoom is veranderd, dus fitness moet opnieuw berekend worden
        child.requires_eval = True

    return population


# ============================================================================ #
#  4. FITNESS
# ============================================================================ #

def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:

    return mean_plus_std_tree_edit_distance(body, targets)

# ============================================================================ #
#  5. EVALUATION FUNCTION
# ============================================================================ #

def evaluate(population: Population, targets: list[nx.DiGraph]):

    for individual in population.unevaluated:
        body = individual.genotype.to_networkx()
        individual.fitness = fitness_function(body, targets)

    return population

# ============================================================================ #
#  6. SELECTION
# ============================================================================ #


def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= config.target_population_size:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]

        if ind_a.fitness_ is None or ind_b.fitness_ is None:
            raise ValueError("Fitness missing")
        
        if ind_a.fitness_ <= ind_b.fitness_:
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population

# ============================================================================ #
#  7. LOOKING AT A BODY
# ============================================================================ #


def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


# ============================================================================ #
#  8. ENTRY POINT
# ============================================================================ #


def main() -> None:
    targets = load_targets()
    population_size = 80
    config.target_population_size = population_size 

    population = Population(
        [make_individual() for _ in range(population_size)]
    )

    population = evaluate(population, targets)
    population = parent_selection(population)
    population = crossover(population)
    population = evaluate(population, targets)
    population = survivor_selection(population)

    parents = population.where(
        lambda ind: bool(ind.tags.get("selected", False))
    )

    children = population.where(
        lambda ind: bool(ind.tags.get("mutate", False))
    )

    console.log("")
    console.log(f"parents: {len(parents)}")

    for index, parent in enumerate(parents):
        console.log(
            f"parent {index}: fitness={parent.fitness_:.4f}"
        )

    console.log("")
    console.log(f"children: {len(children)}")

    for index, child in enumerate(children):
        console.log(
            f"child {index}: fitness={child.fitness_:.4f}"
        )

if __name__ == "__main__":
    main()
