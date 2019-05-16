# estatic

Code to solve the electrostatic problem with conductors and dielectrics in 2D.

## Install

Just place `estatic2d.py` alongside your script and do
```python
import estatic2d
```
The required modules are `numexpr`, `numpy`, `scipy` and `matplotlib`, which
are included in Anaconda and anyway can be installed at a shell with
```sh
$ pip install numpy scipy matplotlib numexpr
```

## Documentation

The documentation is in the code, start with
```python
>>> import estatic2d
>>> help(estatic2d)
```
or read directly the documentation from the file. Run the scripts `cap_plate.py`, `cap_cylinder.py`, `two_plates.py` for examples.

## Features

* Easy to use.

* The computation is a linear system even with dielectrics.
