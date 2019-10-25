#!/usr/bin/env python3
import click


from src.pytorch_selection import *
pytorch_init()

from src.paths import *
from src.pipeline import *
from src.a01_sem_seg import *

b_threaded = False

exp_class = ExpSemSegPSP_Ensemble_BDD

@click.command()
@click.argument('ensemble_id', type=int)
def main(ensemble_id):

	cfg = add_experiment(exp_class.cfg,
	    name = "{norig}_{i:02d}".format(norig=exp_class.cfg['name'], i=ensemble_id),
	)
	exp = exp_class(cfg)
	print_cfg(cfg)

	exp.init_default_datasets(b_threaded)

	# exp.datasets['val'].set_fake_length(20)
	# exp.datasets['train'].set_fake_length(20)

	exp.init_net("train")
	exp.init_transforms()
	exp.init_loss()
	exp.init_log()
	exp.init_pipelines()
	if b_threaded:
		exp.pipelines['train'].loader_class = SamplerThreaded
		exp.pipelines['val'].loader_class = SamplerThreaded

	exp.training_run()

	#qsub -jc 12h.1gpu /home/lis/programs/uge_run.sh "python3 /home/lis/dev/unknown_dangers/0103_PSP_ensemble_exec.py 1"

main()
