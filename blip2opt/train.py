"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
from attackutils import myGlobal
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from blip2opt.models import *
myGlobal._init()
b = 1e-6
loss_name="DRSL3"
start=0
end=20
myGlobal.set_value('b', b)
myGlobal.set_value('loss_name', loss_name)
myGlobal.set_value('start', start)
myGlobal.set_value('end', end)
print("loss {} b={} start={} end={}".format(loss_name,b,start, end))
pretrain_path='/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL__1/BLIP2/DRSL3_6_0_20/20230829144/checkpoint_0.pth'
pretrain_path='/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_6_0_20/20230821151/checkpoint_0.pth'
all_answers=[]
#myGlobal.set_value("pretrain_path", pretrain_path)

import blip2opt.tasks as tasks
from blip2opt.common.config import Config
from blip2opt.common.dist_utils import get_rank, init_distributed_mode
from blip2opt.common.logger import setup_logger
from blip2opt.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from blip2opt.common.registry import registry
from blip2opt.common.utils import now
# imports modules for registration
from blip2opt.datasets.builders import *
from blip2opt.processors import *
from blip2opt.runners import *
from blip2opt.tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())


    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()

