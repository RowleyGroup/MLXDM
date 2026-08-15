# Water ball reduction

Scripts to take a spherical "ball" of water molecules from a PDB file and
peel it down to smaller and smaller balls by removing the 10 outermost
water molecules at a time, saving each intermediate structure as its own
XYZ file. Pure standard-library Python, no extra packages required.

## Usage

With your own water-ball PDB file (each water molecule as its own residue,
i.e. one O + two H atoms sharing a residue sequence number):

```bash
python3 reduce_water_ball.py --input your_water_ball.pdb --outdir output
```

This writes `output/water_ball_<N>.xyz` for every size `N` reached by
removing molecules 10 at a time (farthest from the ball's center first),
starting from the full ball down to the last full batch of 10.

Options:
* `--step` / `-n`: number of water molecules removed per file (default: 10)
* `--prefix`: filename prefix for the XYZ files (default: `water_ball`)

## Generating a sample water ball

If you don't already have a water-ball PDB, `generate_water_ball.py` builds
one (a cubic lattice of TIP3P-geometry waters, randomly oriented, trimmed to
a sphere):

```bash
python3 generate_water_ball.py --radius 10.0 --output water_ball.pdb
python3 reduce_water_ball.py --input water_ball.pdb --outdir output
```
