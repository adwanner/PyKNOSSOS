# PyKNOSSOS
PyKNOSSOS is a software tool for the visualization and annotation of 3D image data and was developed for high-throughput image annotation of 3D electron microscopy stacks of brain tissue. https://ariadne.ai

Ergonomic software is important to minimize the amount of human labor in circuit reconstruction. PyKNOSSOS is a  software package for manual skeleton tracing, synapse annotation and visualization.
PyKNOSSOS was inspired by KNOSSOS (http://www.knossostool.org).

# Features:
1. Efficient rendering of reconstructed neurons using VTK. Using the visualization toolkit (VTK;
http://www.vtk.org/), PyKNOSSOS can interactively visualize hundreds of reconstructed neurons
in 3D on a standard desktop computer together with the underlying image data. This functionality
is important for efficient data browsing, tracing and error correction. For higher-level analyses,
visualization in PyKNOSSOS can be integrated into custom workflows. Moreover, PyKNOSSOS
can be used to generate 3D renderings and animated displays of raw data and reconstructions.

2. Multi-resolution view. Like KNOSSOS, PyKNOSSOS dynamically loads cubes of data into
RAM as users navigate through a volume. This allows users to browse through large datasets
(>1TB) with minimal RAM requirements. Storing multiple sets of cubes at different resolutions
allows for seamless zooming through large volumes and the extraction of virtual reslices with
arbitrary orientation and zoom-level. The data can be loaded from local hard disk storage, via
a hybrid streaming pipeline from HTTP accessible servers, or via the JPEG stack service of the
data API of neurodata (http://www.neurodata.io).

3. Virtual reslicing of the raw data orthogonal to local processes. Tracing of neurites and branch
point detection should be particularly efficient when users view a section through the EM data
volume that is orthogonal to the process being traced. Such virtual reslices should also facilitate
the identification of synapses because presynaptic vesicles, the synaptic density, and the
postsynapse are contained in the same view. PyKNOSSOS automatically calculates a rotation-minimized
“locally orthogonal” section during tracing and presents it in a separate viewport, in
addition to the cardinal cross-sections and the imaging plane. We found that the “locally
orthogonal” view facilitates the detection of branch points and increases tracing speed.

4. Synapse annotation tools. PyKNOSSOS includes tools to define the location and direction of a
synapse by three successive clicks on the presynaptic process, the synaptic density, and the
postsynaptic process. In addition, synapses can be assigned to user-defined classes by a
confidence level. Furthermore, pre-calculated “flight paths” can be loaded to automatically visit
all branches of a reconstructed neuron. Using this mode, only two clicks on the synaptic density
and postsynapse are required to define a synaptic connection. These features, together with the
“locally orthogonal” view, facilitate manual synapse annotation. Optionally, each synapse can be
assigned to a predefined class and annotated with a confidence level.
