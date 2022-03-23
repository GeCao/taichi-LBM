### How to run code

```commandline
pip install taichi
pip install --upgrade PyMCubes
python main.py
```

### Features:
* LBM
    -[x] 2D basic
    -[x] 3D basic
    -[x] 2D KBC model
    -[ ] 3D KBC model
    -[x] 2D Shan-Chen multiphase model
    -[x] 3D Shan-Chen multiphase model
    -[ ] 2D He-Chen-Zhang multiphase model
    -[x] 3D He-Chen-Zhang multiphase model
    -[x] Obstacle boundary handling
  
![result1](data/doubly-periodic-shear-layer.png) 

(doubly periodic shear layer: case 1)


![result2](data/Shan-Chen-Box.png) (Before)
![result2](data/Shan-Chen-Sphere.png) (After: case 2, 3, 4)