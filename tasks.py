import os
from invoke import Context, task
from google.cloud import secretmanager
import wandb
import time
from google.cloud import secretmanager
import wandb

WINDOWS = os.name == "nt" # this means that the OS is Windows
PROJECT_NAME = "wikipedia"
PYTHON_VERSION = "3.12"

# prefix string to try uv first, then python
#prefix = "uv" if not WINDOWS else "python"

#if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
#    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-37e77e52cd8f.json"

def login_wandb(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    
    project_id = "dtumlops-448012"	
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=name)
    
    secret = response.payload.data.decode('UTF-8')
    os.environ["WANDB_API_KEY"] = secret

    wandb.login()

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
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def trainfast(ctx: Context) -> None:
    """Train model with fast flag."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py --num-splits=1 --num-epochs=10", echo=True, pty=not WINDOWS)

# wandb sweep --project <propject-name> <path-to-config file>
@task
def sweep(ctx: Context) -> None:
    """Run hyperparameter sweep."""
    if os.environ.get("WANDB_API_KEY") == "" or os.environ.get("WANDB_API_KEY") == None:
        login_wandb("WANDB_API_KEY")
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
    
    ctx.run(f"wandb agent mlops2025/wiki_classification/{sweep_id}", echo=True, pty=not WINDOWS)

# docker run --gpus all --rm -it -v "$(pwd)/cloud:/app/cloud:ro" -v "$(pwd)/tasks.py:/app/tasks.py:ro" --entrypoint "uv" train-image-gpu run invoke sweep
# "\\wsl.localhost\Ubuntu\home\fenriswulven\project\wiki-classification-mlops\checkpoints\split_0\best_model-epoch=195-val_acc=0.8219-v4.ckpt"
## test with best model

@task
def visualize(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py", echo=True, pty=not WINDOWS)


@task 
def test2(ctx: Context) -> None:
    """Test model on test set."""
    ctx.run(f"uv run src/{PROJECT_NAME}/test.py", echo=True, pty=not WINDOWS)  

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run pytest tests --cov=src --cov-report=xml", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report", echo=True, pty=not WINDOWS)

@task
def dockerbuild(ctx: Context, progress: str = "plain") -> None:
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
def builddocs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def servedocs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

