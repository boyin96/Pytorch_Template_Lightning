import hydra

from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from typing import List

from src.utils import pylogger

__all__ = [
    "instantiate_callbacks",
    "instantiate_loggers",
]

log = pylogger.RankedLogger(__name__, use_rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiates callbacks from config.
    Args:
        callbacks_cfg: A DictConfig object containing callback configurations.
    Returns:
        A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping...")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_name}>...")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """
    Instantiates loggers from config.
    Args:
        logger_cfg: A DictConfig object containing logger configurations.
    Returns:
        A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_name, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_name}>...")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
