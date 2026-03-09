# Different ways to test the code

## To time using `benchmark.py`

```bash
python3 benchmark.py
```

## To profile using `cProfile`

```bash
python3 -m cProfile -s cumulative volumerender.py
```

## To profile using `line_profiler`

Add the ```@profile``` decorator

```bash
python3 -m kernprof -l volumerender.py
python3 -m line_profiler volumerender.py.lprof
```

## To profile using `memory_profiler`

Add the ```@profile``` decorator

```bash
python3 -m memory_profiler volumerender.py
```

To __visualise__: 

```bash
python3 -m mprof run volumerender.py
python3 -m mprof plot mpro[...].dat
```

## To profile using the `perf-tool`

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 volumerender.py
```

