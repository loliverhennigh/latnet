
# LatNet (Beta Release)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/-lkJwtK4qWU/0.jpg)](https://www.youtube.com/watch?v=-lkJwtK4qWU)

A library for performing Lattice Boltzmann fluid simulations using neural networks. A paper going over the method can be found [here](https://arxiv.org/abs/1705.09036). The original source code for the paper can also be found [here](https://github.com/loliverhennigh/Phy-Net). This library represents an extension of the original work and is designed to handle large scale simulations. This is a beta release and does not contain all the desired features and has untested functionality. The library uses TensorFlow for the neural networks and Sailfish to generate the fluid simulations.

# Getting Started

## Sailfish Library

LatNet uses the [Sailfish](http://sailfish.us.edu.pl/) to generate the fluid simulations. Sailfish requires a GPU and CUDA to run. Installing CUDA can be specific to your system and GPU but we recomend installing it in the way [TensorFlow suggests](https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support). You will also need the following packages,

`
apt-get install python-pycuda python-numpy python-matplotlib python-scipy python-sympy
apt-get install python-zmq python-execnet git
`

To obtain the Sailfish library run,

`
./setup_sailfish.sh
`

There are multiple forks of Sailfish and this version was tested with Python 2.7 and a GTX 1080 GPU with CUDA 8.0. We will test with CUDA 9.0 soon.

## Train Network

To run the network reported in the original paper, move to the example directory and run
`
python standard_network_trainer.py
`

This will generate a train set of simulations and requires around 50 GBs of memory. Training should require around 12 hours. The network is saved in the directory `network_save` and you can moniter progress with tensorboard.

There are many parameters you can change for training such as using Generative Advesarial Loss and multiple GPUs. To get a complete list of parameters run with `--help` flag.

## Evaluate Network

Once the network is trained to resonable performance you evaluate in by running

`
python standard_network_eval.py
`

This will generate a video comparing the the generated simulation to the original in the `figs` directory and save all the data in the `simulation` directory.

# Improvements on Previous Work

* Pressure and Velocity boundary conditions. Forceing term are also implemented but the force needs to be constant for train and test set.

* Generative Advesarial Training as seen in [here](https://arxiv.org/pdf/1801.09710.pdf).

* Arbitrary sized simulations for train set.

* Dynamic Train set allowing simulations to be generated while training network.

* Multi-GPU training.

* Running simulations on CPU ram instead of GPU dram allowing for much larger simulations to be generated (1,000 by 1,000 by 2,000 grid cells with only 8 GB of CPU ram).

* New network architectures.

* New domains such as lid driven cavity and forced simulation.

* Data agumentation such as flipping and rotating simulation.

# Future Imporvements

These are the features with priority that will be released in future versions.

* Active Learning. High priority.

* 3D simulations. Most code already in however will finish adding after all 2D tests complete.

* Multi-GPU evaluating on single server.

* Physicaly inspired network architecture. 

* Add other non-Lattice Boltzmann solvers.


