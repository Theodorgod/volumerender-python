# Original Code

## Results from `benchmark.py`

```bash
$ python3 benchmark.py

Nangles = 10
Average run time over 10 iterations: 18.029819893836976
Standard deviation over 10 iterations: 0.26950125576995604
```

## Results from `cProfile`

```bash
python3 -m cProfile -s cumulative volumerender.py
```

_[Too much information, not whole is printed]_

```bash
$ python3 -m cProfile -s cumulative volumerender.py
True
         2016437 function calls (1977858 primitive calls) in 24.429 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    710/1    0.008    0.000   24.441   24.441 {built-in method builtins.exec}
        1    0.003    0.003   24.441   24.441 volumerender.py:1(<module>)
        1    0.887    0.887   17.451   17.451 volumerender.py:42(main)
       10    0.177    0.018   12.833    1.283 _rgi.py:645(interpn)
       90    0.008    0.000   12.824    0.142 __init__.py:1(<module>)
       10    0.026    0.003   12.632    1.263 _rgi.py:375(__call__)
       10   10.978    1.098   10.980    1.098 _rgi.py:520(_evaluate_linear)
    865/4    0.014    0.000    6.980    1.745 <frozen importlib._bootstrap>:1349(_find_and_load)
    853/4    0.012    0.000    6.979    1.745 <frozen importlib._bootstrap>:1304(_find_and_load_unlocked)
    826/6    0.010    0.000    6.975    1.163 <frozen importlib._bootstrap>:911(_load_unlocked)
  2064/10    0.008    0.000    6.975    0.697 <frozen importlib._bootstrap>:480(_call_with_frames_removed)
    690/6    0.004    0.000    6.974    1.162 <frozen importlib._bootstrap_external>:1017(exec_module)
   629/50    0.002    0.000    5.745    0.115 {built-in method builtins.__import__}
  734/135    0.010    0.000    5.474    0.041 <frozen importlib._bootstrap>:1390(_handle_fromlist)
     1800    3.361    0.002    3.361    0.002 volumerender.py:16(transferFunction)
```

## Results from `line_profiler`

_Observe that comments in the result were removed for readability_

```bash
Timer unit: 1e-06 s

Total time: 31.7104 s
File: volumerender.py
Function: main at line 41

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    41                                           @profile
    42                                           def main():
    43                                               """Volume Rendering"""
    44                                           
    46         1        589.4    589.4      0.0      f = h5.File("datacube.hdf5", "r")
    47         1      17824.6  17824.6      0.1      datacube = np.array(f["density"])
    48                                           
    50         1          8.2      8.2      0.0      Nx, Ny, Nz = datacube.shape
    51         1        148.1    148.1      0.0      x = np.linspace(-Nx / 2, Nx / 2, Nx)
    52         1         28.6     28.6      0.0      y = np.linspace(-Ny / 2, Ny / 2, Ny)
    53         1         20.5     20.5      0.0      z = np.linspace(-Nz / 2, Nz / 2, Nz)
    54         1          1.9      1.9      0.0      points = (x, y, z)
    55                                           
    57         1          2.0      2.0      0.0      images = []
    58                                               
    60         1          2.0      2.0      0.0      Nangles = 10
    61        11         31.4      2.9      0.0      for i in range(Nangles):
    65        10         43.6      4.4      0.0          angle = np.pi / 2 * i / Nangles
    66        10         19.1      1.9      0.0          N = 180
    67        10        574.5     57.5      0.0          c = np.linspace(-N / 2, N / 2, N)
    68        10     202275.6  20227.6      0.6          qx, qy, qz = np.meshgrid(c, c, c)
    69        10       3377.6    337.8      0.0          qxR = qx
    70        10     223821.3  22382.1      0.7          qyR = qy * np.cos(angle) - qz * np.sin(angle)
    71        10     218977.1  21897.7      0.7          qzR = qy * np.sin(angle) + qz * np.cos(angle)
    72        10     192824.4  19282.4      0.6          qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
    73                                           
    75        10   26606358.0 2.66e+06     83.9          camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))
    76                                           
    78        10        738.2     73.8      0.0          image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))
    79                                           
    80      1810      29745.4     16.4      0.1          for dataslice in camera_grid:
    81      1800    3907976.0   2171.1     12.3              r, g, b, a = transferFunction(np.log(dataslice))
    82      1800     108843.2     60.5      0.3              image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
    83      1800      96961.8     53.9      0.3              image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
    84      1800      96567.4     53.6      0.3              image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]
    85                                           
    86        10        682.4     68.2      0.0          image = np.clip(image, 0.0, 1.0)
    87                                                   
    89        10        501.5     50.2      0.0          images.append((image * 255).astype(np.uint8))

   100         1         84.9     84.9      0.0      images_array = np.array(images)
   101         1         49.2     49.2      0.0      if os.path.exists("volumerender_images_new.npy"):
   102         1        174.8    174.8      0.0          os.remove("volumerender_images_new.npy")
   103         1       1165.3   1165.3      0.0      np.save("volumerender_images_new.npy", images_array)
   104                                                                    
   106         1          2.4      2.4      0.0      return 0

```

## Results from `memory_profiler`

```bash
Filename: volumerender.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    41  117.121 MiB  117.121 MiB           1   @profile
    42                                         def main():
    43                                             """Volume Rendering"""
    44                                         
    46  117.938 MiB    0.816 MiB           1       f = h5.File("datacube.hdf5", "r")
    47  182.398 MiB   64.461 MiB           1       datacube = np.array(f["density"])
    48                                         
    50  182.398 MiB    0.000 MiB           1       Nx, Ny, Nz = datacube.shape
    51  182.402 MiB    0.004 MiB           1       x = np.linspace(-Nx / 2, Nx / 2, Nx)
    52  182.402 MiB    0.000 MiB           1       y = np.linspace(-Ny / 2, Ny / 2, Ny)
    53  182.406 MiB    0.004 MiB           1       z = np.linspace(-Nz / 2, Nz / 2, Nz)
    54  182.406 MiB    0.000 MiB           1       points = (x, y, z)
    55                                         
    57  182.406 MiB    0.000 MiB           1       images = []
    58                                             
    60  182.406 MiB    0.000 MiB           1       Nangles = 10
    61  611.594 MiB -212.742 MiB          11       for i in range(Nangles):
    65  611.594 MiB -189.543 MiB          10           angle = np.pi / 2 * i / Nangles
    66  611.594 MiB -189.543 MiB          10           N = 180
    67  611.594 MiB -189.543 MiB          10           c = np.linspace(-N / 2, N / 2, N)
    68  656.090 MiB  344.410 MiB          10           qx, qy, qz = np.meshgrid(c, c, c)
    69  611.594 MiB -590.008 MiB          10           qxR = qx
    70  611.594 MiB -144.570 MiB          10           qyR = qy * np.cos(angle) - qz * np.sin(angle)
    71  611.594 MiB -145.047 MiB          10           qzR = qy * np.sin(angle) + qz * np.cos(angle)
    72  611.594 MiB  -56.059 MiB          10           qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
    73                                         
    75  632.273 MiB  254.344 MiB          10           camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))
    76                                         
    78  632.273 MiB   -0.465 MiB          10           image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))
    79                                         
    80  611.594 MiB -38694.516 MiB        1810           for dataslice in camera_grid:
    81  611.594 MiB -38293.531 MiB        1800               r, g, b, a = transferFunction(np.log(dataslice))
    82  611.594 MiB -38293.590 MiB        1800               image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
    83  611.594 MiB -38293.594 MiB        1800               image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
    84  611.594 MiB -38293.594 MiB        1800               image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]
    85                                         
    86  611.594 MiB -212.742 MiB          10           image = np.clip(image, 0.0, 1.0)
    87                                                 
    89  611.594 MiB -212.742 MiB          10           images.append((image * 255).astype(np.uint8))

   100  588.672 MiB  -22.922 MiB           1       images_array = np.array(images)
   101  588.672 MiB    0.000 MiB           1       if os.path.exists("volumerender_images_new.npy"):
   102  588.672 MiB    0.000 MiB           1           os.remove("volumerender_images_new.npy")
   103  588.676 MiB    0.004 MiB           1       np.save("volumerender_images_new.npy", images_array)
   104                                         
   105                                         
   106  588.676 MiB    0.000 MiB           1       return 0
```

## To profile using the `perf-tool`

```bash
$ perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 volumerender.py

 Performance counter stats for 'python3 volumerender.py':

    78,496,318,055      cycles:u                         #    3.819 GHz                         (83.32%)
   124,505,721,335      instructions:u                   #    1.59  insn per cycle              (83.34%)
     8,077,673,656      cache-references:u               #  392.971 M/sec                       (83.32%)
       677,105,750      cache-misses:u                   #    8.38% of all cache refs           (83.33%)
    23,348,824,885      branches:u                       #    1.136 G/sec                       (83.33%)
        34,785,883      branch-misses:u                  #    0.15% of all branches             (83.35%)
    20,555,377,514      task-clock:u                     #    1.126 CPUs utilized             
           356,653      faults:u                         #   17.351 K/sec                     
           356,653      minor-faults:u                   #   17.351 K/sec                     
                 0      cs:u                             #    0.000 /sec                      
                 0      migrations:u                     #    0.000 /sec                      

      18.256422428 seconds time elapsed

      17.582219000 seconds user
       2.985410000 seconds sys
```






