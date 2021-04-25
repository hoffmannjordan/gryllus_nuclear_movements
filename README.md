## Code accompanying _Local density determines nuclear movements during syncytial blastoderm formation in a cricket_
# [bioRxiv Link]()
`REPOSITORY IS IN PROGRESS. CHECK BACK SOON.`

By [Seth Donoughe](https://www.sethdonoughe.com/), [Jordan Hoffmann](https://jhoffmann.org/), [Taro Nakamura](http://www.nibb.ac.jp/niimilab/), [Christopher H. Rycroft](https://people.seas.harvard.edu/~chr/), and [Cassandra Extavour](https://www.extavourlab.com/)

![gryllus_sim](./ims/gryllus_sim.png)

## Code to Simulate Embryos
### Simulation Dependencies

The code uses `numpy` (v. 1.18.1) and `scipy` (v. 1.4.1). Additionally, we use `scikit-fmm` (v. 2019.1.30) [GitHub Link](https://github.com/scikit-fmm/scikit-fmm) [PyPI Link](https://pypi.org/project/scikit-fmm/). The instal time is just a few minutes.

### Code Output

The code outputs `csv` files that contain positions through time and the time since the last division. 
The code can also output individual shells or an auxilliary file useful for tracking the fate of individual nuclei. 
When nuclei divide, one daughter takes the index of the original nucleus, and another is added at the end. 
The expected run time for 800 time steps is approximately 8 hours.

### Potential Code Modifications
<!-- `IN PROGRESS. CHECK BACK SOON`

* How to change the geometry.

* How to change the shell size.

* How to plot the results. -->

* How to render the shells.

See [here](https://github.com/hoffmannjordan/Insect-Development-Model) for a _Mathematica_ notebook that can be used to generate a [POV-ray](http://www.povray.org/) file.

## Data
The `dataset` folder contains 200 tracked timepoints with a total of 12,864 nucleus timepoints. 

## Additional Utilities
### `ilastik` to `MaMuT` 
[Mathematica](https://www.wolfram.com/mathematica/) code to convert [ilastik](https://www.ilastik.org/) tracks to [MaMuT](https://imagej.net/MaMuT) for manual correction is included in `Convert_Ilastik_to_Mamut.nb`. 
This code has not been tested with the most recent version of either `ilastik` or `MaMuT`, since it is now deprecated. 
The functionality has been folded in to `ilastik` and `MaMuT` (it was not when we did this work), but we recommend that users use the supported functionaltiy.