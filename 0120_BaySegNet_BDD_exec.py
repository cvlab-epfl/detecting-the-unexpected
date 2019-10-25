#!/usr/bin/env python3

from src.pytorch_selection import *
pytorch_init()

from src.paths import *
from src.pipeline import *
from src.a01_sem_seg import *

b_threaded = False

exp = ExpSemSegBayes_BDD()

exp.init_default_datasets(b_threaded)

# exp.datasets['val'].set_fake_length(10)
# exp.datasets['train'].set_fake_length(10)

exp.init_net("train")
exp.init_transforms()
exp.init_loss()
exp.init_log()
exp.init_pipelines()
if b_threaded:
	exp.pipelines['train'].loader_class = SamplerThreaded
	exp.pipelines['val'].loader_class = SamplerThreaded

exp.training_run()

#qsub -jc 24h.1gpu /home/lis/programs/uge_run.sh "python /home/lis/dev/unknown_dangers/0109_EpistemicPSP_exec.py"

