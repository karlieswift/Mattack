"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from blip2opt.common.registry import registry
from blip2opt.tasks.base_task import BaseTask
from blip2opt.tasks.captioning import CaptionTask
from blip2opt.tasks.image_text_pretrain import ImageTextPretrainTask
from blip2opt.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from blip2opt.tasks.retrieval import RetrievalTask
from blip2opt.tasks.vqa import VQATask, GQATask, AOKVQATask
from blip2opt.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from blip2opt.tasks.dialogue import DialogueTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
]
