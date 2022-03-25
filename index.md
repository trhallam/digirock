# `digirock`

`digirock` is a Python framework for modelling digital rock models. It uses four
abstract building blocks `Element`s, `Blend`ers, `Transform`ers and `Switch`es
 to build flexible and moduluar rock models, all on top of the Python scientific
 stack to make things easy (`numpy`, `xarray` and `pandas`).

- The [`Element`](api/base_classes.md#digirockelement) class is the fundamental building block, all
   other classes a based on it and it provides most of the functionality. Element
   classes are extended to have rock or fluid properties or indeed any other property
   you want to model.
- The [`Blend`](api/base_classes.md#digirockblend) class is used to combine any of the building block classes
   in different ways
   the blending method is implemented within a new class. Examples of a blend method
   include a Wood's Fluid or Voight-Reuss-Hill average for a mineral composite.
- The [`Transform`](api/base_classes.md#digirocktransform) class is a pipeline that takes one of the
  building block classes and performs an operation on either the upgoing or downgoing
  properties of the rock. This is useful for transforming inputs or performing adjustments
  to lower level block outputs. For example, Nur's Critical porosity transform is a
  transformer.
- The [`Switch`](api/base_classes.md#digirockswitch) class is used when you zones or regions in your data
  that use completely different models. The switch class for example can be used apply
  the correct fluid in different PVT Zones, or a different Rock model per facies zone.

`digirock` has primarily been implemented for clastic petro-elastic modelling but the
framework is flexible enough for users to implement there own Elements and Models
following the guidelines and examples from withing this documentation. If you write a
new model, or implement an old one, please considering submitting back to the `digirock`
project ([contributing](contrib.md)).

## Quick Start

See the quick start example in the user guide.

## Installation

### Installing with `pip`

`digirock` is available via `pip install`.

```
pip install digirock
```

### Installing from source

Clone the repository

```
git clone http://github.com/trhallam/digirock
```

and install using `pip`

```
cd digirock
pip install .
```
