# Rethinking Operation Reserach: learning on-demand delivery policies with supervised deep learning

Repository of the Semester Project: Rethinking Operation Reserach: learning on-demand delivery policies with supervised deep learning. 

It contains the final report, slides from the final presentation and code. 

The folder `Codes /Python` contains the code of the project which includes:

* Jupyter Notebooks with different results:

	*  `00-Data Exploration.ipynb`: data exploration from the initial dataset
	*  `01-Graphical representation.ipynb`: explore different planar representations of the problem and the influence of several parameters.
	*  `02-Generate Input.ipynb`: first try of self-developed simulator to reproduce events to evalute results. 
	*  `03-Simulation.ipynb`: compared results of differnet models based on simulator times. 
	*  `07.x - Policy... .ipynb`: different notebooks where different neural network models are trained and evaluated. 
	* `10-final-net.ipynb`: train and evaluation of the the set of final selected models.
	* `10.5-Save_models.ipynb`: train and save final models.
	* `11-Time_evaluation.ipynb`: evaluation in terms of time of different models using the simulator.
	* `12-Overall_time.ipynb`: statistics of comparison between different models and target policy.
	* `OSMNX-SF network.ipynb`: notebook exploring the representation of the porblem using Open Street Maps

* `/functions`: directory that contains several `.py` files used for the simulator, models training as well as final evaluation. The files contain different functions in each case. To execute the notebooks, these files should be placed in the same directory of the notebooks. 

* `/minmax_data`: data to train models and generate events
* `/model_weights` and `/model_weights2`: directory containing the weights of different trained models. The models used in the evaluations of the final models are in the directory 2. 
* `/other_attempts`: directory that contains different notebooks with some other models or simulator explorations that were discarded for the final analysis. 
* `/tables`: initial data 


The environment used for the project can be duplicated by running the following command in the terminal: 

```conda create --name <envname> --file requirements.txt```


For questions regarding the code, environment installation or any other details, please contact me at: natalie.bolonbrun@epfl.ch 

January, 2020.

Natalie Bolón Brun


