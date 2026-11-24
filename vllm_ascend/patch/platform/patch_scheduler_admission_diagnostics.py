# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging

from vllm.v1.core.sched import scheduler as scheduler_module

from vllm_ascend import envs
from vllm_ascend.core.scheduler_admission_diagnostics import install_scheduler_admission_diagnostics

logger = logging.getLogger(__name__)

# BalanceScheduler replaces the public name. Its upstream base is also the
# parent of RecomputeScheduler, ProfilingChunkScheduler and AsyncScheduler.
# Instrument its constructor, not a copied schedule() implementation.
_BASE_SCHEDULER = next(
    (
        cls
        for cls in scheduler_module.Scheduler.__mro__
        if cls.__module__ == "vllm.v1.core.sched.scheduler" and cls.__name__ == "Scheduler"
    ),
    None,
)
if _BASE_SCHEDULER is None:
    logger.warning("SchedulerAdmission diagnostics NOT installed: upstream Scheduler base was not found.")
else:
    install_scheduler_admission_diagnostics(
        _BASE_SCHEDULER,
        envs.VLLM_ASCEND_SCHEDULER_ADMISSION_LOG_INTERVAL_MS / 1000.0,
    )
    logger.info("SchedulerAdmission diagnostics enabled (rate-limited, scheduler CPU observations only).")
