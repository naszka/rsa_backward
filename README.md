Learning in the RSA model backwards
============

	1. install packages from rsa_backward/requirements.txt

	2. Put ./amortized_rsa on your PYTHONPATH
	~~~
	export PYTHONPATH=/path/to/here/amortized_rsa:$PYTHONPATH
	~~~

	This contains a clone of https://github.com/juliaiwhite/amortized-rsa 
	with minimal changes to accomodate newer python versions.

	We use this as an external package for the base Listener implementation.

	3. The main code for the paper is in ./rsa_backward
	~~~
	cd rsa_backward
	~~~

Listener and speaker levels
------------

> Important! The speaker and listener levels use different numbering in the paper and in the code base.
> code level --> paper level

    * speaker 1 --> speaker 1
    * speaker 2 --> speaker 3
    * listener 0 --> listener 0
    * listener 1 --> listener 2
    * listener 2 --> listener 4


Training models
------------

Train models with all possible environment parameters of level 0

~~~
python train_models.py --listener_level 0
~~~

This will save the the best models under .models/ ,
and different checkpoints in subfolders models/epoch_1, models/epoch_5, models/epoch_10 .
 
To alter the environment parameters you have to edit the file train_models.py .

~~~python
# correlation parameter for P(S|S)
max_alphas = (1, 10)
# number of distractor images
num_distractors_params = (2, 3, 4)
# levels of speakers that the model learns from
speaker_levels = (1, 2)
~~~
 
Evaluation reasoning learners
--------------

Evaluate all models with the the same level they were trained with in all possible environments.

~~~
python evaluate_listeners.py
~~~

We did not include all possible trained models in this package. But you can find trained models under .models/epoch_10 to play with.

The packaged models were trained with

* num_distractors=2
* speaker_level=1
* cost=0.6
* maxalpha=1

maxalpha=1 indicates uniform shape distribution.

Results are saved under results/rsa_results.ljson as linewise json.

You can change the environment variables at the top of the file.
The file evaluates all models trained in all possible setups with all possible evaluation environments.

Evaluation upgraded listeners
--------------

Upgrade level 0 listeners to higher levels and evaluate them in all environments.

~~~
evaluate_listeners_0_upgraded.py
~~~

Results are saved results/rsa_results_0_upgraded.ljson
