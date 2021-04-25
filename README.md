## Supporting code for "Local density determines nuclear movements during syncytial blastoderm formation in a cricket"

By [Seth Donoughe](https://www.sethdonoughe.com/), [Jordan Hoffmann](https://jhoffmann.org/), [Taro Nakamura](http://www.nibb.ac.jp/niimilab/), [Chris H. Rycroft](https://people.seas.harvard.edu/~chr/), and [Cassandra G. Extavour](https://www.extavourlab.com/)

![gryllus_sim](./ims/gryllus_sim.png)

## Code to simulate pulling clouds that move nuclei

### Installation and usage
Install these dependencies:
- `numpy` (v. 1.18.1)
- `scipy` (v. 1.4.1)
- `scikit-fmm` (v. 2019.1.30)
    - See [GitHub link](https://github.com/scikit-fmm/scikit-fmm) or [PyPI Link](https://pypi.org/project/scikit-fmm/).

The install time is a few minutes. To run a simulation, run `sim.py`.

### Code output
The code outputs `csv` files of nuclear positions through time and each nucleus's time since last division. The code can also output individual clouds or an auxilliary file useful for tracking the fate of individual nuclei. When nuclei divide, one daughter is assigned the index of the original nucleus, and another daughter is assigned an index at the end. The expected run time for 800 time steps is approximately 8 hours on a typical desktop computer.

### Additional info
<!-- `IN PROGRESS. CHECK BACK SOON`

* How to change the geometry.

* How to change the shell size.

* How to plot the results. -->

**Render the output clouds:** See [here](https://github.com/hoffmannjordan/Insect-Development-Model) for a _Mathematica_ notebook that can be used to generate a [POV-ray](http://www.povray.org/) file.

## Example empirical dataset
The `dataset` folder contains 200 tracked timepoints from a 3D+T lightsheet dataset of a preblastoderm _G. bimaculatus_ embryo with a total of 12,864 nucleus-timepoints.

## Additional utility
We used [Mathematica](https://www.wolfram.com/mathematica/) code to transfer [ilastik](https://www.ilastik.org/) tracks to [MaMuT](https://imagej.net/MaMuT). For the purposes of reproducibility, we include this utility as `Convert_Ilastik_to_Mamut.nb`. This code has not been tested with the most recent version of either `ilastik` or `MaMuT`. It is now deprecated because its functionality has been added to `ilastik` and `MaMuT` directly.
