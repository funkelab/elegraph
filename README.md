## Elegraph

### TODOs

- [ ] Generate 2D simulated worm based on sampling from specific distribution
- [ ] Visualize such generated worms
- [ ] Learn the transformation parameters using max likelihood approach
- [ ] Ensure that the generated points do not encroach within existing worm
- [ ] Extend this analysis to 3D
- [ ] Apply to real data
- [ ] Visualize worms 


### Create Environment

```
mamba create -n elegraph python
mamba activate elegraph
git clone https://github.com/funkelab/elegraph
git checkout likelihood
cd elegraph
pip install -e .
```

