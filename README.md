# Code Appendix
The code of the paper

## Requirements to run the code:

Python 3.6-3.10 Windows x64

Numpy

pytorch

cvxopt

## Basic usage
Copy one line of commands in the same folder of `./gfedplat` and run (one example shown as follows).

```
python run.py --seed 1 --device 0 --module CNN --algorithm FedAvg --dataloader DataLoader_cifar10_dir --N 100 --Diralpha 0.1 --B 50 --C 0.1 --R 3000 --E 1 --lr 0.1 --decay 0.999
```

All parameters can be seen in `./gfedplat/main.py`.

By setting different parameters and run the command, you can replicate results of all experiments.

Enjoy yourself!

Paper Hash code:
7caba0708a2666d7e86c64bae8d64ab6

Appendix Hash code:
394c1c375852d9cc6a7107b5a3b80332

For the safety of the code, some files are encrypted, but they can be run directly. They will be public when the paper is published.
