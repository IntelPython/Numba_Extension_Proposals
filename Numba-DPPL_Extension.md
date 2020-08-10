# Section 1. Introduction

This proposal has the high-level implementation design of a new Numba extension called `numba-dppl`. The `numba-dppl` extension will introduce 
additional decorators to the Numba front-end and a back-end code-generator to automatically offload data-parallel kernels to SYCL devices.
Aspects of the front-end extensions are discussed separately (refer [language design proposal](https://github.com/IntelPython/Numba_Extension_Proposals/blob/master/Language_Design_Proposal.md)).

`Numba-dppl` was designed as a standalone Numba extension to minimizes, as much as possible, changes to the core codebase of Numba. In Section 2, we demonstrate at a
high-level how our implementation works. We would like to note that in Numba's current design not all of our changes can be outlined into a separate module. 
For all such places, we want to discuss additional extension points for Numba. Section 3 has a list of currently identified cases where we cannot easily 
separate out our changes into an extension module.

# Section 2. A new Lowerer for `Parfor` Nodes for Automatic GPU offloading

## Section 2.1. Current state of the art

Numba provides [`lowering.lower_extensions`](https://github.com/numba/numba/blob/56fc9d7eeb098002753c13480bcde72dcfe0296c/numba/parfors/parfor_lowering.py#L483)
in `numba/parfors/parfor_lowering.py` that allows registering new types of "lowerers" for special Numba IR nodes. The current CPU lowerer for parfor nodes
uses this extension point. 

```python
lowering.lower_extensions[parfor.Parfor] = _lower_parfor_parallel
```

Please refer [register other extensions via entry points](http://numba.pydata.org/numba-doc/latest/extending/entrypoints.html) for more details about Numba extension points.

## Section 2.2. Registering the new `numba-dppl` extension module

The `numba-dppl` extension will provide and register the following functionalities:

* new lowering function for `Parfor` nodes,
* the `dppl.kernel` and other decorators to do explicit OpenCL kernel-style
  programming, and
* a new set of compiler passes and pipeline to support the new lowerer.

To use the new decorators in `numba-dppl`, users will need to explicitly import the `numba-dppl` package.

## Section 2.3. Adding a Lowering fallback to default `Parfor` lowerer

The new lowerer included in `numba-dppl` will support falling back onto the default Numba `Parfor` lowerer to handle cases where `numba-dppl` does not yet
support GPU kernel generation. A high-level implementation of the fallback behavior will look as follows:

```python
from numba import lowering
from numba.parfors import parfor
import dppl

def _lower_parfor_gufunc(lowerer, parfor):
    if dppl.get_context():
        <do offloading to GPU>
    else:
        _lower_parfor_parallel(lowerer, parfor)

lowering.lower_extensions[parfor.Parfor] = _lower_parfor_gufunc
```

The `dppl.get_context` is an API function provided by a companion `dppl` package that indicates if the `@jit` function call originated inside a
`dppl.device_context` scope (refer [Language Design RFC](https://numba.discourse.group/t/rfc-language-design-for-a-new-back-end-to-automatically-offload-data-parallel-kernels/164) for details).

Note that the above behavior is only triggered when the `numba-dppl` extension is in use.

<!---
### Section 2.1.4. Lowering fallback (Ver.2)

Context manager could change `lowering.lower_extensions[parfor.Parfor]`.
Context manager provides sets its custom lowering function at enter and return
it to the previous state on leave.
--->

## Section 2.4. Device context implementation

As described in the [language design](https://github.com/IntelPython/Numba_Extension_Proposals/blob/master/Language_Design_Proposal.md)
a dppl `device_context` configuration as the only way to automatically generate device kernels and to 
generate the host code needed for running the kernel. Our implementation of the offloading inside the new lowerer will rely on `dppl` to store as a global
variable the current context configuration. Numba will access the global state by use prebound functions in the `dppl` module.

Inside `dppl` the `device_context` configuration state is maintained as a stack data structure to support nested contexts. A `dppl` context manager pushes the
specified `device_context` configuration to the stack on entering the context manager scope and then pops the configuration from the top of the stack on
exiting the scope. `numba-dppl` will rely on `dppl` to get a handle for the current `device_context`. This `dppl` getter function returns the current
context from the top of the stack or returns a default context (i.e. CPU) when the stack is empty.

# Section 3. Possible New Numba Extension Points

For implementing the `numba-dppl` extension we need to extend Numba in some additional places that are not yet exposed using extension points. The following
is a list of places that we have presently identified that will need modifications in addition to introducing the new lowerer.

1. Function Dispatch

    Numba uses only the function signature to look up the cached versions of a
    `@jit` decorated function. In addition to signature, we will need
    `device_context` to also becomes part of key storing versions of a
    previously compiled function.

2. Caching (across program executions)

    Numba's feature to serialize JIT compiled functions for usage across
    multiple execution of a program only supports caching the wrapper and
    outer-native code. We want to extend the feature to support serializing the
    compiled kernels generated by our custom lowerer. Also, like function
    dispatch, the device context would need to be in the key here.

3. Pass Manager

    `numba-dppl` has a custom pass pipeline that reuses most of the default
    Numba pass pipeline. Instead, we will want to discuss adding registration
    points in Numba's pass pipeline that allows us to load our passes into
    Numba's pipeline at specific points, such as after type-inferencing, without
    having to replicate the whole of the pipeline.

<!---
4. Existing passes extending - depends on pass

Scenario: extend only part of Parfor pass, for example not fusing parfors that
can run on GPU with those that can't.

5. jit parameters - not possible now

Scenario: imagine ParallelAccelerator is an extension. Then parallel is foreign
parameter. The same with selecting offloading device via 
parallel={offload: 'gpu'} parameter.
--->

