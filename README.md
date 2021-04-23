## Code accompanying _Local density determines nuclear movements during syncytial blastoderm formation in a cricket_
# [bioRxiv Link]()

By [Seth Donoughe](https://www.sethdonoughe.com/), [Jordan Hoffmann](https://jhoffmann.org/), [Taro Nakamura](http://www.nibb.ac.jp/niimilab/), [Christopher H. Rycroft](https://people.seas.harvard.edu/~chr/), and [Cassandra Extavour](https://www.extavourlab.com/)

### Requirements

The code uses `numpy` and `scipy`. Additionally, we use `scikit-fmm` [GitHub Link](https://github.com/scikit-fmm/scikit-fmm) [PyPI Link](https://pypi.org/project/scikit-fmm/).

### Code Output

The code outputs `csv` files that contain positions through time and the time since the last division. 
The code can also output individual shells or an auxilliary file useful for tracking the fate of individual nuclei. 
When nuclei divide, one daughter takes the index of the original nucleus, and another is added at the end. 

### Change the Geometry