import os
from invoke import Context, task
import time

WINDOWS = os.name == "nt" # this means that the OS is Windows
PROJECT_NAME = "wikipedia"
PYTHON_VERSION = "3.12"

# prefix string to try uv first, then python
#prefix = "uv" if not WINDOWS else "python"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

#python src/wikipedia/train.py mlops-proj-group3-bucket torch_geometric_data
@task
def train_cloud(ctx: Context) -> None:
    """Train model on Google Cloud AI Platform."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py mlops-proj-group3-bucket torch_geometric_data", echo=True, pty=not WINDOWS)

# wandb sweep --project <propject-name> <path-to-config file>
@task
def sweep(ctx: Context) -> None:
    """Run hyperparameter sweep."""
    # Initialize the sweep and get the Sweep ID
    result = ctx.run("wandb sweep configs/sweep/sweep.yaml", echo=True, pty=not WINDOWS)
    
    #Extract the Sweep ID from the command output
    sweep_id = None
    for line in result.stdout.splitlines():
       if "Run sweep agent with:" in line:
           sweep_id = line.split("wandb agent ")[-1].strip()
           sweep_id = sweep_id.split("/")[-1]
           sweep_id = sweep_id[:8]
           break
    if not sweep_id:
       raise RuntimeError("Sweep ID could not be determined. Check the output of `wandb sweep`.")
    
    ctx.run(f"wandb agent {sweep_id}", echo=True, pty=not WINDOWS)

# "\\wsl.localhost\Ubuntu\home\fenriswulven\project\wiki-classification-mlops\checkpoints\split_0\best_model-epoch=195-val_acc=0.8219-v4.ckpt"
## test with best model
@task 
def test2(ctx: Context) -> None:
    """Test model on test set."""
    ctx.run(f"uv run src/{PROJECT_NAME}/test.py models/best_model-epoch=195-val_acc=0.8219-v4.ckpt", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

