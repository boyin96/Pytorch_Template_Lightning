import warnings

from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
from omegaconf import DictConfig
from functools import wraps

from src.utils import pylogger, rich_utils

__all__ = [
    "extras",
    "task_wrapper",
    "get_metric_value",
]

log = pylogger.RankedLogger(__name__, use_rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """
    Applies optional utilities before the task is started.
     Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing
    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # Return if no "extras" config.
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # Disable python warnings.
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=true>")
        warnings.filterwarnings("ignore")

    # Prompt user to input tags from command line if none are provided in the config.
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=true>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # Pretty print config tree using Rich library.
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=true>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """
    Optional decorator that controls the failure behavior when executing the task function.
    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)
    Args:
        task_func: The task function to be wrapped.
    Returns:
        The wrapped task function.
    """
    @wraps(task_func)
    def wrapper(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Execute the task.
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # Things to do if exception occurs.
        except Exception as ex:
            # Save exception to .log file.
            log.exception("")

            # Some hyperparameter combinations might be invalid or cause out-of-memory errors.
            # So when using hparam search plugins like Optuna, you might want to disable.
            # Raising the below exception to avoid multirun failure.
            raise ex

        # Things to always do after either success or exception.
        finally:
            # Display output dir path in terminal.
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # Always close wandb run (even if exception occurs so multirun won't fail).
            if find_spec("wandb"):  # Check if wandb is installed.
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict
    return wrapper


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """
    Safely retrieves value of the metric logged in LightningModule.
    Args:
        metric_dict: A dict containing metric values.
        metric_name: If provided, the name of the metric to retrieve.
    Returns:
        If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
