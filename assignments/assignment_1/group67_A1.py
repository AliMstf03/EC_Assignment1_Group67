# Assignment 1 - Group 67
# import the functions that are already maade in the example 
from ariel.ec import EA, EAOperation, Individual, Population, config
import csv

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
POPULATION_SIZE = 80
GENERATIONS = 100
SEEDS = [42, 43, 44, 45, 46]

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
            

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================ #
#  8. ENTRY POINT
# ============================================================================ #

def run_ea(targets, mutation_function, seed, variant_name):
    set_seed(seed)

    population_size = 80
    generations = 100

    config.target_population_size = population_size

    # Initial population
    population = Population(
        [make_individual() for _ in range(population_size)]
    )

    population = evaluate(population, targets)

    results = []

    # Generation 0
    fitness_values = [
        ind.fitness_
        for ind in population.alive
        if ind.fitness_ is not None
    ]

    results.append({
        "variant": variant_name,
        "seed": seed,
        "generation": 0,
        "best_fitness": min(fitness_values),
        "mean_fitness": float(np.mean(fitness_values)),
    })

    # Evolution
    for generation in range(1, generations + 1):

        # Only keep survivors from previous generation
        population = population.alive

        # Parent selection
        population = parent_selection(population)

        # Crossover
        population = crossover(population)

        # The only difference between the two EA variants
        population = mutation_function(population)

        # Evaluate new children
        population = evaluate(population, targets)

        # Bring population back to target size
        population = survivor_selection(population)

        # Fitness statistics
        fitness_values = [
            ind.fitness_
            for ind in population.alive
            if ind.fitness_ is not None
        ]

        results.append({
            "variant": variant_name,
            "seed": seed,
            "generation": generation,
            "best_fitness": min(fitness_values),
            "mean_fitness": float(np.mean(fitness_values)),
        })

    return results

### Dit is de algoritme 
def run_ea(targets, mutation_function, seed, variant_name):
    set_seed(seed)

    population_size = 80
    generations = 100

    config.target_population_size = population_size

    # Initial population
    population = Population(
        [make_individual() for _ in range(population_size)]
    )

    population = evaluate(population, targets)

    results = []

    # Generation 0
    fitness_values = [
        ind.fitness_
        for ind in population.alive
        if ind.fitness_ is not None
    ]

    results.append({
        "variant": variant_name,
        "seed": seed,
        "generation": 0,
        "best_fitness": min(fitness_values),
        "mean_fitness": float(np.mean(fitness_values)),
    })

    # Evolution
    for generation in range(1, generations + 1):

        # Only keep survivors from previous generation
        population = population.alive

        # Parent selection
        population = parent_selection(population)

        # Crossover
        population = crossover(population)

        # The only difference between the two EA variants
        population = mutation_function(population)

        # Evaluate new children
        population = evaluate(population, targets)

        # Bring population back to target size
        population = survivor_selection(population)

        # Fitness statistics
        fitness_values = [
            ind.fitness_
            for ind in population.alive
            if ind.fitness_ is not None
        ]

        results.append({
            "variant": variant_name,
            "seed": seed,
            "generation": generation,
            "best_fitness": min(fitness_values),
            "mean_fitness": float(np.mean(fitness_values)),
        })

    return results

def run_random_search(targets, seed):
    set_seed(seed)

    population_size = 80
    generations = 100
    samples_per_generation = population_size // 2

    results = []

    best_so_far = float("inf")

    for generation in range(generations + 1):

        if generation == 0:
            number_to_generate = population_size
        else:
            number_to_generate = samples_per_generation

        random_population = Population(
            [make_individual() for _ in range(number_to_generate)]
        )

        random_population = evaluate(
            random_population,
            targets
        )

        fitness_values = [
            ind.fitness_
            for ind in random_population
            if ind.fitness_ is not None
        ]

        generation_best = min(fitness_values)

        best_so_far = min(
            best_so_far,
            generation_best
        )

        results.append({
            "variant": "Random Search",
            "seed": seed,
            "generation": generation,
            "best_fitness": best_so_far,
            "mean_fitness": float(np.mean(fitness_values)),
        })

    return results


def main():
    targets = load_targets()

    seeds = [42, 43, 44, 45, 46]

    all_results = []

    for seed in seeds:

        console.log(f"Running seed {seed}")

        # Variant 1: Point mutation
        console.log("Point mutation")
        point_results = run_ea(
            targets,
            mutation_point,
            seed,
            "Point Mutation"
        )

        all_results.extend(point_results)

        # Variant 2: Subtree mutation
        console.log("Subtree mutation")
        subtree_results = run_ea(
            targets,
            mutation_subtree,
            seed,
            "Subtree Mutation"
        )

        all_results.extend(subtree_results)

        # Random-search baseline
        console.log("Random search")
        random_results = run_random_search(
            targets,
            seed
        )

        all_results.extend(random_results)

    # Save raw experimental results
    output_file = DATA / "experiment_results.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "variant",
                "seed",
                "generation",
                "best_fitness",
                "mean_fitness",
            ],
        )

        writer.writeheader()
        writer.writerows(all_results)

    console.log(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()