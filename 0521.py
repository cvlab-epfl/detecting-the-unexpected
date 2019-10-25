#!/usr/bin/env python3

from src.pytorch_selection import *
pytorch_init()

from src.a05_differences import *

exp = Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT()
exp.init_default_datasets()

exp.init_net("train")
exp.init_transforms()
exp.init_loss()
exp.init_log()
exp.init_pipelines()
exp.training_run()

#qsub -jc 24h.1gpu /home/lis/programs/uge_run.sh "python /home/lis/dev/unknown_dangers/0502b_CorrDiff_fakeErr_refine_exec.py"

