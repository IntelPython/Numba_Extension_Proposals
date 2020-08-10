# Section 1. Introduction

In this proposal, we introduce a new backend to Numba with an automatic offload
optimizer for data-parallel kernels detected inside a @jit decorated function.
The proposal only discusses offloading for SYCL [[1](https://www.khronos.org/sycl/)]
devices and other types of devices such as CUDA may be considered at a later
point. The new automatic offload optimizer bears similarities with Numba's
existing ParallelAccelerator optimizer, and shares the "parfor" analysis passes
with ParallelAccelerator. Similar to ParallelAccelerator, the new optimizer
identifies data-parallel code regions or parfors inside a @jit function and only
operates on those parfors. Each parfor that is identified by our optimizer gets
offloaded as a SYCL kernel to a specific SYCL device. Section 2 goes into the
details about our proposed code generation approach.

Since a @jit decorated function can have multiple parfors, a function may have
multiple kernels. The backend automates the host-side control code to enqueue
the kernels, move data from host to device, and finally to synchronize across
kernel boundaries. The host-side code will do the to-and-fro copying of data
between Numba's internal representation for NumPy ndarrays to a memory region
that can be accessed from a SYCL device. Section 3 of this proposal has the
details about handling host-device data movement. In addition to offloading data
parallel parfor kernels, a future extension to our backend may incorporate
data-flow analyses to identify sequential code regions that can also be
offloaded as SYCL single tasks. Such an extension will be primarily incorporated
to minimize host-device data movement.

Finally, as part of the proposal we introduce a dependency on Intel's new
PyDPPL package [[2](https://github.com/IntelPython/pydppl)]. PyDPPL is a
lightweight Python package that provides a minimal wrapper for SYCL and OpenCL
runtime classes to Python. The new optimizer relies on PyDPPL to acquire SYCL
queues, enqueue kernels on a queue, create SYCL buffers, and allocate SYCL USM
memory.

# Section 2. Code Generation

Code portability was one of the driving motivations for the language design and
code-generation design of the new backend. Our goal is to let existing Numba
programs take advantage of automatic offloading of parfors with minimal
change to existing @jit functions. We do not add any new arguments to the
@jit decorator and propose a PyDPPL context manager triggered code generation
approach to enable automatic offloading. Listing 2.1 is a basic example showing
the new proposed usage. In the example, the function "foo" defined on Line 5
defaults to normal Numba semantics when called on Line 9. For a user to trigger
automatic offloading, the function must be invoked explicitly from within a
PyDPPL "device_context". Once invoked inside a PyDPPL device_context Numba uses
the new backend and offloads any parfors identified inside foo to the device
specified as the argument to the PyDPPL context manager.

### Listing 2.1
```
1 import dppl
2 from numba import jit
3
4 @jit
5 def foo():
6     ...
7
8 # Invoking foo outside a PyDPPL context results in the default Numba behavior
9 foo()
10
11 with dppl.device_context(gpu):
12    # Triggers GPU code generation
13    foo()
14
15 with dppl.device_context(cpu):
16    # Triggers code generation for a SYCL CPU device
17    foo()
```

## Section 2.1 Caching of @jit functions

The new language design also changes the way Numba caches JIT compiled
functions. Currently Numba uses only the function signature to cache versions of
a previously JIT compiled function. Since, our backend uses a PyDPPL
device_context for specializing the code-generation process we will extend
Numba's caching policy to be aware of the PyDPPL device_context, if any, that
was used when JIT compiling a function. Listing 2.1 illustrates the modified
caching behavior; the JIT compiled version of foo on Lines 9, 13, and 17 are
all different and would be stored as different versions of foo in the cache.

It was a conscious decision on our part to not add any new arguments to the
Numba @jit decorator. Additional @jit decorator arguments may lead to
scenarios where a device_context may be specified on the decorator and also
using a context manager leading to potential confusion on the part of a
programmer in understanding which device_context option takes precedence over
the other. We leave it to the programmer to define features that make it
possible to invoke a @jit decorated function without having to explicitly create
a context manager scope. Listing 2.2 shows one way of implementing such a
feature.

### Listing 2.2
```
1 import numba as nb
2
3 def gpujit(func):
4   def wrp_gpujit(*args, **kwargs):
5     jitted = nb.jit()(func)
5     if dppl.get_context():
6       return jitted(*args, **args)
7     else:
8       with dppl.device_context(gpu):
9         jitted(*args, **kwargs)
10    return wrp_gpujit
11
12 @gpujit
13 def foo(a):
14    ...
```

## Section 2.2 What Code Regions are Automatically Offloaded

When a @jit function is invoked from within a dppy.device_context region,
JIT compilation of the function is triggered with the automatic addition of
Numba's parallel=True option. This option is described in the Numba
documentation and lists all the Python operations that Numba recognizes as
having parallel semantics. It is these parallel operations that form the code
regions that are then automatically offloaded as SYCL kernels. Currently, all
such operations identified as having parallel semantics are parallelized on
multi-core and the same will initially hold true for SYCL offload but over time
a cost model may be introduced to prevent small parallel regions that would not
benefit from offload from being converted into SYCL kernels and then offloaded.

## Section 2.3 Handling @jit arguments

As stated in Section 2.2, the new backend defaults to generating parallel SYCL
kernels. This code-generation strategy may at times conflict with explicit
options passed by a programmer to the @jit decorator. Listing 2.2 shows an
example where the programmer has explicitly set the parallel option for the
function foo to False. The presumable intent of the programmer is to execute the
foo serially on Line 8. It is unclear then what the programmer intends to
happen on Line 12. The programmer has apparently selected two contradictory
options: by setting the parallel argument to False in the @jit decorator he/she
indicated serial code-generation, and by calling foo from inside a PyDPPL
device_context context manager he/she is indicating they want to turn on
automatic offloading that inherently generates data-parallel code.

### Listing 2.3
```
1  import dppl
2  from numba import jit
3
4  @jit(parallel = False)
5  def foo():
6      pass
7
8  foo()
9
10 with dppl.device_context(gpu):
11     # triggers GPU code generation
12     foo()
```

To handle this type of scenarios, we propose the following:

1. The new backend throws an error as "parallel=False" conflicts with
   automatic offloading.
2. The new backend ignores all @jit flags that do not apply to it. In
   Listing 2.3, that will imply that Numba ignores the original
   "parallel=False" argument and proceeds with attempting to offload
   parfors as SYCL kernels.
3. The new backend gives precedence to the original arguments passed to the
   decorator. In which case, Numba turns off detection and offloading of
   parfors. However, if the backend identifies scenarios where it is
   beneficial to execute some code region on a device, albeit serially, it
   is free to do so.
4. The new backend may also optionally choose to execute parfors as SYCL
   single tasks.

We propose adopting (1) as the default behavior.

## Section 2.4 Scheduling of Kernels

At present, the kernels generated for parfors operate on all available
work items or threads. Thus, the number of work items per dimension of the data
is equal to the size of that dimension. The size of each dimension is passed
unchanged as the global work size, in OpenCL terminology, for that dimension to
the underlying kernel invocation mechanism. We presently allow the underlying
OpenCL runtime to determine the number of work items per work group, i.e., the
local size in OpenCL terminology. More advanced scheduling techniques may be
adopted at a later point.

All automatically offloaded kernels will execute synchronously. A design to 
support asynchronous kernel execution based on SYCL events and/or accessors is
under consideration and will be proposed in a future proposal.

# Section 3. Data Movement and Synchronization

This section discusses the automatic handling of host-device data movement by
the backend.

### Listing 3.1
```
1  import numpy as np
2  from numba import jit
3
4  # x,y are ndarrays representing 1D vectors
5  @jit
6  def foo(x, y):
7      a = 0.5
8      some_serial_preprocessing(x)
9      # Data parallel region 1
10     y = a*x + y
11     # End of data parallel region 1
12     another_serial_processing(y)
13     # Data parallel region 2
14     r = np.sum(y)
15     # End of data parallel region 2
16     return r
```

The memory model for the generated code is to guarantee that the result of any
computation on a device is available on the host before its next use on the
host. At a very basic level the guarantee can imply that data movement and
synchronization happens at the boundary of each parfor. Listing 3.1 shows a
simple scenario that highlights when data is copied to a device and brought
back to the host. The function foo has two parfors: the "axpy" operation on
Line 10 and the reduction operation on Line 14. The two parfors cannot be fused
due to the function call on Line 12. For scenarios such as Listing 3.1,
our backend would treat each parfor as a single unit of synchronization. Data is
moved to the device on entry of the parfor and brought back to the host on exit.

Now let us consider the scenario in Listing 3.2. In this example, the "print"
function call on Line 12 once again prevents fusion of the parfors. However, the
output of the first parfor need not be moved to the host, as it is not used by
the print function on Line 12. The memory model does not preclude scenarios
where compiler analyses may determine that data movement need not happen at the
parfor boundary. For such situations, the compiler is free to explore
optimization opportunities that reduce excess data movement and delay the copy
of data back to the host.

### Listing 3.2
```
1 import numpy as np
2 from numba import jit
3
4 # x,y are ndarrays representing 1D vectors
5 @jit
6 def foo(x, y):
7    a = 0.5
8    some_serial_preprocessing(x)
9    # Data parallel region 1
10   y = a*x + y
11   # End of data parallel region 1
12   print("I do not need anything from the device")
13   # Data parallel region 2
14   r = np.sum(y)
15   # End of data parallel region 2
16   return r
```

Before returning from a @jit function all data is made host accessible. Note
that we use the term host accessible to indicate that the data is accessible on
the host before its next usage. The behavior can be to explicitly copy back data
to the host inside a NumPy array, or in the case of special data containers that
are backed by SYCL’s USM shared memory
[[3](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc)],
we eschew explicit copy back and let
the USM allocator copy the data back as needed. It is an implementation-specific
detail that we omit in this language design proposal.

## Section 3.1 Data Movement Optimization Across @jit functions

We now discuss various use cases where data can be kept device accessible across
multiple @jit decorated functions. The use cases demonstrate the usage of
a SYCL USM shared allocator backed array container that we call "DPArray". Let us
assume that DPArray sub-classes NumPy's ndarray and supports all ndarray
operations automatically, with the execution happening on the CPU. Our new
backend can support such a container the same way as a regular ndarray. When the
backend detects that a container uses SYCL's USM shared allocator it will not
generate any data movement code. Instead, the backend will rely on the SYCL USM
shared allocator to handle data movement. The kernel code will also be changed
to reflect the usage of a USM-based container. The next few Listings illustrate
various scenarios that demonstrate potential usage of a container such as
DPArray.

### Listing 3.3
```
1  import dparray as np
3  import dppl
4  from numba import jit
5
6  @jit
7  def foo(a):
8     a[:] = np.sin(a)
9
10  a = np.ones(100)
11 # on CPU
12 foo(a)
13
14 with dppl.device_context(gpu):
15    # on GPU
16    foo(a)
```

In Listing 3.3, the data of the DPArray "a" is made available on the device
by the USM allocator after foo is called on Line 16. The data is still
accessible on the device even after foo returns on Line 16.

### Listing 3.4
```
1 import dparray as np
3 import dppl
4 from numba import jit
5
6  @jit
7  def foo(a):
8     a[:] = np.sin(a)
9     b = a + 2
10    return b
11
12 a = np.ones(100)
13
14 # on CPU
15 foo(a)
16
17 with dppl.device_context(gpu):
18    # on GPU
19    b = foo(a)
```

In Listing 3.4, the data of the DPArray "a" is accessible on the device after
foo is called on Line 19. The DPArray "a" remains accessible on the device even
after foo returns. The DPArray "b" returned by foo is also device accessible on
Line 19.

### Listing 3.5
```
1 import dparray
2 import numpy as np
3 import dppl
4 from numba import jit
5
6 @jit
7 def foo(a, b):
8    a[:] = np.sin(a)
9    b[:] = a + 2
10
11 a = dparray.ones(100)
12 b = dparray.ones(100)
13
14 # on CPU
15 foo(a,b)
16
17 with dppl.device_context(gpu):
18    # on GPU
19    foo(a, b)
```

In Listing 3.5, both the arguments are passed in as DPArrays and they remain
accessible on the device even after foo returns.

### Listing 3.6
```
1 import dparray
2 import numpy as np
3 import dppl
4 from numba import jit
5
6 @jit
7 def foo(a, b):
8    a[:] = np.sin(a)
9    b[:] = a + 2
10   some_serial_operation(b)
11
12 a = dparray.ones(100)
13 b = dparray.ones(100)
14
15 # on CPU
16 foo(a,b)
17
18 with dppl.device_context(gpu):
19    # on GPU
20    foo(a, b)
```

In Listing 3.6, the serial operation on Line 10 cannot be offloaded. Therefore,
the operation is executed on the CPU, and USM handles moving the data back to
the host. Do note that moving the data back to the host does not necessarily
invalidate the device copy of the data. If the CPU operation was a read-only
operation, then the device data for array "b" can still be used on the device.
Thus, on returning on Line 20 the array "a" is on the device, the array "b" has
been copied back to the host, but the device copy may still be usable depending
on the operation on Line 10.

# Section 4. Conclusion

The current version of the proposal discussed Intel's plans to extend Numba by
adding an automatic offload feature by extending Numba's existing
ParallelAccelerator optimizer. Multiple use cases covering both code
generation and data movement across host and device are included with this
proposal. Subsequent revisions can be made to the proposal after getting
feedback from the Numba open-source community.


# References

1. [https://www.khronos.org/sycl/](https://www.khronos.org/sycl/)
2. [https://github.com/IntelPython/pydppl](https://github.com/IntelPython/pydppl)
3. [https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc)
