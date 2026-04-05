**DEFINITION OF PARAMETERS IN PROCESSING LAYERS:**

Based on the literature of cognitive neurosciences and the connectionist
paradigm, the relevant parameters of depth perception in the visual
system of the healthy adult human that will be adapted to the artificial
cognitive model are divided into 27 phases, listed in the following
table:

  -------- ----------------------------- -------- ------------------------------------
  **\#**   **Phase**                     **\#**   **Phase**

  **1**    *Light entering the pupil*    **15**   *Visual cortex V1 (Layer I)*

  **2**    *Refraction in cornea and     **16**   *Visual cortex V2 (thick bands)*
           lens*                                  

  **3**    *Projection on the retina*    **17**   *Visual cortex V2 (thin bands)*

  **4**    *Transformation into nerve    **18**   *Visual cortex V2 (Intercalated
           impulses*                              bands)*

  **5**    *Optic chiasma*               **19**   *Visual cortex V3 (Binocular
                                                  Disparity Integration)*

  **6**    *Optical strips*              **20**   *Visual cortex V3 (Relative Motion
                                                  Analysis)*

  **7**    *Lateral geniculate nucleus   **21**   *Visual cortex V3 (Three-dimensional
           (LGN)*                                 shapes)*

  **8**    *Visual cortex V1 (Layer      **22**   *Visual cortex V4 (Color
           IV-Cα)*                                Processing)*

  **9**    *Visual cortex V1 (Layer      **23**   *Visual cortex V4 (Complex Shape
           IV-Cβ)*                                Processing)*

  **10**   *Visual cortex V1 (Layer      **24**   *Visual cortex V4 (Visual Attention
           IV-B)*                                 Processing)*

  **11**   *Visual cortex V1 (Layer      **25**   *Visual cortex V5 (MT) (Motion
           IV-A)*                                 Analysis)*

  **12**   *Visual cortex V1 (Layer II   **26**   *Visual cortex V5 (MT) (Binocular
           and III)*                              Disparity)*

  **13**   *Visual cortex V1 (Layer V)*  **27**   *Visual cortex V5 (MT) (Construction
                                                  of Dynamic Spatial Maps)*

  **14**   *Visual cortex V1 (Layer VI)*          
  -------- ----------------------------- -------- ------------------------------------

These phases are presented below, declaring the computational processing
corresponding to each one:

*Phase 1: Light enters the pupil.*

In the human visual system, the pupil dynamically regulates the amount
of light entering the eye, adapting to different lighting conditions to
optimize visual perception. This process must be simulated using an
initial preprocessing of the images from the *KITTI dataset.* This
preprocessing must represent the independent entry of light into each
eye, which implies implementing two input channels that process the left
and right images separately, thus simulating human binocular reality.

The goal of this phase is to simulate pupil function by performing
operations that adjust the dynamic range and contrast of images,
replicating the visual system\'s ability to adapt to different levels of
illumination. To achieve this, intensity normalization, histogram
matching, and light noise reduction must be performed.

First, the intensity of the input images must be normalized by scaling
the pixel values between 0 and 1. This process ensures the
standardization of visual information and allows simulating the
adaptation of the pupil to varying lighting conditions. The mathematical
formula used for this operation is:

$$I_{normalizado} = \frac{I - min(I)}{(I)\  - min(I)}$$

Where 𝐼 represents the intensity values of the original image. This step
ensures that the values fall within a consistent range, facilitating
further processing.

Next, histogram equalization must be applied to simulate the increase in
contrast in the image, emulating the effect of the visual system under
suboptimal lighting conditions. This process redistributes pixel
intensities to maximize the utilization of the available dynamic range.
Finally, a Gaussian filter must be implemented to reduce light noise and
smooth out abrupt intensity transitions, replicating the optical
system\'s function of removing high-frequency artifacts.

Since visual information enters the system from both eyes, it is
necessary to perform this processing independently for the left and
right images of the *KITTI dataset.*

Finally, it is important to validate this phase through automated and
visual testing. Automated testing should verify that the dynamic range
of pixels is within the expected values, while visual testing should
confirm that the processed images retain relevant details and present
improvements in contrast compared to the originals.

*Phase 2: Refraction in cornea and lens*

This process ensures that light rays are properly focused on the retina.
The cornea performs most of the refraction, while the lens dynamically
adjusts the focus for objects at different distances. In the
computational model, this phase must simulate the joint action of these
two optical structures to ensure that visual data is accurately
processed in the subsequent phases.

The objective of this phase is to transform the images preprocessed in
Phase 1, which represent light entering through the pupil, to simulate
the optical refraction that occurs naturally in the eye. To achieve
this, operations must be applied that simulate the general smoothing
provided by the cornea and the fine adjustment performed by the lens.
This computational processing will allow the sharpness and convergence
of the images to be optimized, ensuring that they are ready to be
processed in retinal projection in the next phase.

The computational model must perform this processing in two main stages.

The first stage involves the application of a Gaussian filter that
emulates the action of the cornea by softening and redirecting light
rays toward a common focal point. This softening ensures that most of
the light energy is concentrated on relevant areas of the image.

The second stage is based on an adaptive approach that simulates the
dynamic adjustment of the lens, by using a Laplacian filter to enhance
edges and improve local image sharpness. This approach ensures that
details of the visual scene are maintained, even under complex lighting
or texture conditions.

Validation of this phase should be done through visual and quantitative
testing. Visual testing will allow comparing images before and after
processing, confirming that the outputs present improved focus.
Quantitative testing, meanwhile, should measure metrics such as local
contrast and sharpness to ensure that the images meet the expected
characteristics after refraction simulation.

Finally, the output of this phase will be sequentially integrated with
the following stages of the model. The processed images will be
projected onto the retina in Phase 3, where the simulation of visual
transduction will begin. This approach ensures the continuity and
consistency of the model, aligning with the neurophysiological
principles described in this work.

*Phase 3: Projection on the retina*

In the human visual system, this stage transforms the light refracted by
the cornea and lens into a topographic representation on the retinal
surface, organized according to the density of retinal receptors. These
receptors, the rods and cones, have distinct functions that are critical
for visual perception. Cones predominate in the fovea and specialize in
the perception of color and detail under bright lighting conditions.
Rods, on the other hand, concentrated in the retinal periphery, are
responsible for sensitivity to luminance and motion, functioning mainly
in low light conditions.

The goal of this phase is to transform the images preprocessed in Phase
2 into a representation that emulates both the retinotopic distribution
and the specific functions of the rods and cones. To achieve this, three
main actions must be implemented. First, a retinotopic transformation is
applied that simulates the variable density of retinal receptors,
increasing the resolution at the center of the image (fovea) and
decreasing it towards the periphery. This transformation ensures that
the visual representation respects the spatial organization of the human
retina.

Second, the information is separated into channels representing the
responses of the cones and rods. The cone channel emulates the
perception of color and detail, using smoothing to extract
high-resolution information. On the other hand, the rod channel focuses
on motion and luminance detection, applying a Laplacian filter that
highlights abrupt changes in the image. This separation allows the model
to reflect the specific functional characteristics of these
photoreceptors.

Finally, each channel is divided into nasal and temporal regions, which
prepares the visual signals for processing at the optic chiasma (Phase
5). Discrimination between regions is performed by dividing each image
into two vertical halves: the left half of the image corresponds to the
nasal region of the right eye and the temporal region of the left eye,
while the right half corresponds to the temporal region of the right eye
and the nasal region of the left eye.

The output of this phase consists of eight matrices for each eye: four
for the cones (nasal and temporal regions) and four for the rods (nasal
and temporal regions). These matrices provide an organized visual
representation that will be used in the transformation into nerve
impulses (Phase 4) and in the crossing of fibers in the optic chiasma
(Phase 5).

*Phase 4: Transformation into nerve impulses*

During this stage, photoreceptors (cones and rods) convert visual
information into electrical signals that are processed and modulated by
bipolar, horizontal, amacrine, and ganglion cells before being
transmitted to the optic nerve. This processing includes critical
mechanisms such as lateral inhibition, which highlights contrasts and
edges, and the organization of ganglion receptive field responses into
*on -center* and *off-center patterns.*

In the computational model, this phase transforms the matrices generated
in Phase 3 (visual representation differentiated by photoreceptor type
and retinal region) into simulated neural signals. The goal is to model
the interactions between the different types of retinal cells and create
maps that emulate the specific responses of ganglion cells, capturing
the most relevant aspects of the visual stimulus, such as local
contrasts and motion.

The implementation of this phase is divided into three main steps.
First, a simulation of visual transduction is applied, which converts
the light intensity of images into a non-linear response, similar to the
behavior of rods and cones. Then, a lateral inhibition network is used
to model the interactions between bipolar and horizontal cells, which
highlights edges and improves contrast perception. Finally, ganglion
response maps are generated, organized in *on-center* and *off-center
patterns,* reflecting the properties of the receptive fields of these
cells.

Validation of this phase should include both visual and quantitative
testing. Visual testing should confirm that the nodal maps highlight
edges and contrasts in a manner consistent with the characteristics of
the visual stimulus. Quantitative testing should measure the
relationship between *on-center* and *off-center* responses, ensuring
that they correctly represent the spatial distribution of retinal
receptive fields.

The output of this phase consists of ganglion maps differentiated by
photoreceptor type (cones and rods) and by response (on *-center* and
*off-center)*. These maps constitute the simulated visual signal that
will be transmitted through the optic nerve, ready for processing in the
optic chiasma (Phase 5).

*Phase 5: Optic chiasma*

In the human visual system, this process is characterized by partial
fiber crossing: fibers from the nasal regions of the retinas (processing
the temporal part of the visual field) cross over to the contralateral
hemisphere, while fibers from the temporal regions of the retinas
(processing the nasal part of the visual field) remain in the same
hemisphere (ipsilateral). This crossing ensures that information from
each half of the visual field is processed in the opposite hemisphere, a
crucial step for binocular integration and depth perception.

In the computational model, this phase should simulate the partial
crossover of visual signals based on the nasal and temporal retinal
regions generated in Phase 4. The ganglionic arrays (on-center and
*off-center)* of rods and cones should be reorganized according to this
crossover structure, ensuring that the signals are correctly distributed
for further processing in subcortical areas, such as the Lateral
Geniculate Nucleus (Phase 7).

The goal of this phase is to implement a routing mechanism that
reorganizes visual signals according to the topology of the optic
chiasma. This process ensures that matrices from the nasal regions of
both eyes are sent to the opposite hemisphere, while matrices from the
temporal regions continue ipsilaterally. The output must maintain the
differentiation between rods and cones, as well as between on center and
off-center responses.

Validation of this phase must confirm that the signals are reorganized
correctly. This includes verifying that the nasal arrays of both eyes
are assigned to the opposite hemisphere, while the temporal arrays
remain in the same hemisphere. In addition, it must be ensured that the
structure and characteristics of the visual data are preserved after the
crossover.

The output of this phase will be the input to the Lateral Geniculate
Nucleus, where a more detailed separation between the magnocellular and
parvocellular pathways will take place, ensuring that the visual
information processed in the optic chiasma is ready for integration in
later phases.

*Phase 6: Optical strips*

In the human visual system, this phase involves the grouping of retinal
fibers into optic tracts and their conduction from the optic chiasma to
the lateral geniculate nucleus (LGN). Fibers from the nasal and temporal
retina of each eye, after being reorganized by hemispheres, are grouped
into two optic tracts: a left and a right one. These tracts carry the
information intact from the optic chiasma to the LGN, where more
advanced processing begins.

In the computational model, this phase simulates the grouping and
routing of visual information reorganized in Phase 5. Although no
additional processing is performed, it is essential to include this
stage to reflect biological continuity and maintain the organized
structure of the signals. The intention is for the optical strips to
preserve the differentiation of information according to the types of
photoreceptors (cones and rods) and the patterns of ganglion response
(on *-center* and *off-center)*, respecting the organization by
hemispheres.

To implement this phase, the rod and cone signals generated in Phase 5
are grouped into two main structures representing the left and right
optic bands. These structures remain intact during transmission to the
LGN, ensuring that the functional organization of the data is preserved.

Validation at this stage is based on ensuring that the signals are
grouped correctly and that their internal structure remains intact
during transmission. Although no additional processing is performed at
this stage, the output of the optical stripes should faithfully reflect
the topographic organization generated in the optic chiasma.

The output of this phase consists of two main structures representing
the left and right optic tracts, ready to enter the Lateral Geniculate
Nucleus in Phase 7. This design ensures that the computational model
respects the neurophysiological sequence and maintains functional
continuity between phases.

*Phase 7: Lateral Geniculate Nucleus (LGN)*

The lateral geniculate nucleus (LGN) is one of the most important
subcortical structures in the human visual system, being the main relay
between the retina and the primary visual cortex (V1). Its function is
not limited to transmitting information, but it also performs active
processing, fine-tuning visual signals to highlight relevant aspects
such as edges, movement, contrast and color. In addition, it integrates
recurrent signals from cortical areas (V1 and V2), allowing it to
dynamically adjust its output based on the visual context, selective
attention and other cognitive demands.

The LGN is composed of six main layers, organized into two magnocellular
(M), four parvocellular (P) and a set of interlaminar zones
(koniocellular, K). These layers have functional specializations:
magnocellular layers mainly process motion and luminance information,
parvocellular layers are responsible for color perception and fine
details, while koniocellular zones complement chromatic processing. In
the computational model, this phase aims to realistically simulate the
flow of information through these layers, respecting the functional
segregation and structural organization of the LGN.

The goal of this phase is to implement a model that reproduces key LGN
operations, including edge and contrast regulation, adjustment of speed
and direction of movement, color processing, and integration of
recurrent signals from the cortex. Interneurons in the LGN are
responsible for adjusting edge sharpness and modulating excitatory
activity, while excitatory relay neurons generate rapid responses to
significant changes in the visual stimulus. In the computational model,
these functions are emulated using deep learning techniques, such as
convolutional networks, which allow filter weights to be dynamically
adjusted to capture these interactions.

For the computational implementation, first, a regulation of edges and
contrast is proposed. Mathematically, the activity of the interneurons
can be represented by a lateral inhibition model that modulates the
signal based on the activity of the neighboring pixels:

$$I_{afinado} = I_{entrada} - \sum_{j \in v(i)}^{}{}w_{ij} \bullet I_{j}$$

where *I ~entered~* is the intensity of the pixel 𝑖, *v(i)* represents
the neighboring pixels and *w ~ij~* They are weights adjusted by the
neural network to optimize contrast.

Subsequently, an adjustment of speed and direction of movement will be
formulated, representing the processing of motion information carried
out by the magnocellular layers, using a convolutional network that
optimizes directions and speeds:

$$v = \frac{\arg\arg\ \int_{}^{}(x,d)\ \ }{d}$$

where 𝑥 are the extracted visual features, 𝑑 are the possible directions
and velocities, and 𝑓 (𝑥 , 𝑑 ) is the probability assigned to each
direction by the network.

Finally, cortical feedback will be integrated, using a model of the
recurrent signals from V1 and V2 that are combined with the original LGN
output to dynamically adjust the signals:

$$I_{modificado} = I_{salida} + \  \propto \  \bullet F_{feedback}$$

where 𝐼 ~output~ is the original signal generated by the LGN, 𝐹
~feedback~ is the recurrent signal from the cortex, and 𝛼 is a parameter
that regulates the influence of the feedback.

The output of this phase consists of signals organized into two main
routes: the magnocellular pathway, which processes motion and luminance
information, and the parvocellular pathway, which processes color and
detail information. These signals will be ready to enter the
corresponding layers of V1 (IV-Cα and IV-Cβ) in the next phase, ensuring
that the functional structure of the visual system is maintained.

*Phase 8: V1 -- Layer IV-Cα*

Layer IV-Cα of the primary visual cortex (V1) is a critical region in
visual processing, particularly for the magnocellular pathway. Its
function is focused on the analysis of rapid motion and luminance
contrasts, operating with a speed and precision that allow the visual
system to react to dynamic stimuli in the environment. This layer
receives signals from the magnocellular layers of the Lateral Geniculate
Nucleus (LGN), processes them intensively, and sends the resulting
information to the higher layers of V1 and other cortical areas, mainly
to IV-B, which continues with the analysis of motion. In addition, IV-Cα
receives feedback from the intercalated bands of V2, which dynamically
modulates and adjusts its activity based on the visual context.

Neurophysiologically, IV-Cα is composed mainly of stellate spiny neurons
and some pyramidal neurons, with moderately sized somas. These neurons
present a radial dendritic arrangement and predominantly excitatory
synaptic activity. Their sensitivity is directed towards luminance
contrasts and rapid movement, which makes them key elements in the
dynamic perception of the environment.

In the computational model, this phase seeks to emulate these
neurophysiological features through a neural network that simulates the
specific processes of IV-Cα. The input of the Phase are the signals from
the magnocellular layers of the LGN, organized according to motion and
luminance coming from Phase 7. The processing involves a refinement of
fast motion and temporal contrasts through convolutional networks and
cortical feedback. A deep learning algorithm will be implemented that
tunes the model parameters to process motion and high temporal contrasts
efficiently. This model will be built on principles of deep
convolutional networks (CNNs ), integrating a *feedback* module to
receive recurrent signals from V2 and dynamically adjusting the weights
based on the context.

The mathematical modeling of layer IV-Cα is divided into three moments,
starting with the processing of motion or the response to dynamic
stimuli that is modeled as the convolution of the visual signal with a
kernel sensitive to specific directions:

$$R(x,t) = \sum_{i = 1}^{N}{}w_{i} \bullet g(x - x_{i},t - t_{i})$$

where 𝑅 (𝑥 , 𝑡 ) is the neural response at position x and time 𝑡 , 𝑔 ( 𝑥
, 𝑡 ) is the motion detection kernel , tuned through training, and 𝑤 ~𝑖~
are the learned weights representing the directional sensitivity.

The processing of high temporal contrasts is then modeled using a
temporal filter that enhances rapid changes in luminance:

$$C(t) = \frac{dI(t)}{dt}$$

where 𝐶 (𝑡 ) is the temporal contrast and 𝐼 ( 𝑡 ) is the luminous
intensity at time 𝑡 .

Finally, feedback integration, where recurrent signals from interleaved
V2 bands adjust the output by a weighted mechanism:

$$O_{modificado} = O_{original} + \  \propto \  \bullet F_{feedback}$$

where ~modified~ O is the adjusted output, 𝐹 ~feedback~ is the recurring
signal, and 𝛼 is a weighting factor that regulates the influence of the
*feedback.*

The validation of this phase is based on evaluating how the network
processes dynamic stimuli, verifying the accurate detection of fast
motion and contrasts, analyzing how V2 *feedback* modulates IV-Cα
outputs in different visual contexts, and evaluating deep learning
metrics such as loss, accuracy, and directional sensitivity.

The output of this phase consists of signals organized into two main
routes: to IV-B of V1 for continued motion analysis and to the higher
layers of V1 for integration with other visual pathways.

*Phase 9: V1 -- Layer IV-C β*

Layer IV-Cβ of the primary visual cortex (V1) is a key region in the
processing of visual information related to fine spatial details and
chromatic differences, features that arrive from the parvocellular
layers of the Lateral Geniculate Nucleus (LGN). This layer specializes
in high-resolution perception and color discrimination, contributing
significantly to the visual system\'s ability to recognize complex
patterns and detect chromatic contrasts in the environment.

Neurophysiologically, layer IV-Cβ is composed primarily of stellate
spiny neurons, which have small and specialized receptive fields. These
neurons have small to moderately sized somas, and their radial dendritic
arrangement facilitates the integration of local signals with
predominantly excitatory synaptic activity. This organization allows
IV-Cβ to respond accurately to visual stimuli with spatial and chromatic
detail. In addition, this layer receives feedback from the thin bands of
V2, which dynamically modulate its activity, adjusting it based on
visual context and attention.

In the computational model, we seek to emulate these properties by using
deep convolutional networks (CNNs ), designed to capture high-resolution
visual patterns and discriminate chromatic differences in the input
signals. This model includes a cortical feedback mechanism that
dynamically adjusts the IV-Cβ outputs, allowing the model to adapt to
varying visual conditions and emulate the contextual integration that
occurs biologically.

The mathematical modeling of layer IV-Cβ starts with the processing of
fine spatial details, which is modeled using high-resolution
convolutions with small kernels, which allow capturing local differences
in visual signals:

$$D(x) = \sum_{i = 1}^{N}{}w_{i} \bullet g(x - x_{i})$$

where 𝐷 (𝑥 ) is the response to details at position 𝑥, 𝑔 ( ⋅ ) is a
Gaussian kernel emphasizing local information, and 𝑤 ~𝑖~ are the weights
learned during training.

Chromatic processing continues through weighted linear combinations of
color channels, adjusted to maximize chromatic discrimination:

$$C_{dif} = \  \propto \  \bullet R + \beta \bullet G + \gamma \bullet B$$

where 𝐶 ~diff~ is the output that highlights chromatic differences and
𝛼, 𝛽 , 𝛾 are parameters that are optimized in training to reflect
chromatic sensitivity

It culminates with the recurrent feedback mechanism that integrates
signals from V2 and dynamically modulates the outputs of IV-Cβ, allowing
the model to prioritize relevant information based on the visual
context:

$$O_{modificado} = O_{original} + \  \propto \  \bullet F_{feedback}$$

where ~modified~ O is the adjusted output, 𝐹 ~feedback~ is the recurring
signal, and 𝛼 is a weighting factor that regulates the influence of the
*feedback.*

Model validation includes visual testing to verify accurate detection of
chromatic differences and fine details, functional testing to evaluate
how V2 *feedback* dynamically modulates IV-Cβ outputs, and performance
testing to measure metrics such as chromatic detection loss and
accuracy.

The information flow in this phase begins with input from parvocellular
signals from the LGN, organized according to spatial details and
chromatic differences. The information is then processed by
convolutional networks that refine details and highlight chromatic
contrasts, integrating recurrent signals from V2 to fine-tune
processing. Finally, the output from IV-Cβ is directed to the higher
layers of V1 for integration with other visual pathways and to the thin
bands of V2, where advanced color and shape analysis is performed.

*Phase 10: V1 -- Layer IV-B*

Layer IV-B of the primary visual cortex (V1) is a region in the
magnocellular pathway specialized in advanced processing of motion,
including direction and speed. This layer receives signals processed in
IV-Cα, which it refines and organizes to send to higher areas of V1 and
to the secondary visual cortex (V2). Layer IV-B is essential for
integrating and expanding the dynamic perception of the visual
environment, allowing the system to interpret moving stimuli with
greater precision.

From a neurophysiological perspective, IV-B is composed of moderately
sized pyramidal and stellate spiny neurons. These neurons present
oriented receptive fields that make them sensitive to the direction of
movement. Their synaptic activity is predominantly excitatory, with a
dendritic pattern that facilitates the integration of signals from
IV-Cα. In addition, IV-B receives cortical feedback from V2 (thick
bands), which dynamically modulates its output based on contextual
demands.

In the computational model, this phase seeks to emulate these
neurophysiological features by using deep convolutional neural networks
(CNNs ) adapted to detect and classify the direction and speed of
movement. A feedback component will be integrated to adjust the outputs
of IV-B based on the recurrent signals coming from V2, aligning with the
biological model.

Regarding mathematical modeling, in layer IV-B the model detects the
direction of movement by applying specialized convolutional filters that
capture spatial and temporal gradients:

$$M_{dirección} = \arg\arg\ \left( \sum_{i}^{}{}w_{i,\theta} \bullet g_{\theta}(x_{i}) \right)\ \ $$

where 𝑀 ~direction~ is the estimated direction of motion, 𝜃 represents
the possible orientations of motion, 𝑔 ~𝜃~ (𝑥 ) is a direction-oriented
filter, and 𝑤 ~𝑖~ , 𝜃 are weights learned during training.

Velocity is then modeled by evaluating changes in stimulus position over
time:

$$v = \frac{\mathrm{\Delta}x}{\mathrm{\Delta}t}$$

where 𝑣 is the velocity of the movement, Δ 𝑥 is the spatial displacement
of the stimulus and Δ 𝑡 is the time interval between samples.

Finally, for cortical feedback integration, recurrent signals from V2
(thick bands) adjust the model outputs through a weighting mechanism:

$$O_{modificado} = O_{original} + \ \beta\  \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ is the feedback
of V2 and 𝛽 is a parameter that regulates the influence of the feedback.

Model validation includes visual testing to confirm accurate detection
of motion directions and velocities in visual stimuli, functional
testing to analyze how V2 feedback modulates IV-B outputs, and
performance testing that measures metrics such as accuracy and loss in
classifying motion directions.

The information flow in this phase begins with input signals processed
in IV-Cα, which contain data related to motion and luminance. These
signals are refined in IV-B, where motion directions and speeds are
classified, integrating feedback from V2 to adjust outputs based on
context. Finally, the processed information is directed to the higher
layers of V1 for integration with other visual pathways and to the thick
bands of V2, where advanced motion analysis is performed.

*Phase 11: V1 -- Layer IV-A*

Layer IV-A of the primary visual cortex (V1) plays an integrative role
in the visual system, receiving signals from both the magnocellular and
parvocellular layers of the Lateral Geniculate Nucleus (LGN). In this
layer, visual information is consolidated in preparation for
transmission to the higher layers of V1. Its function is more generalist
compared to layers IV-Cα and IV-Cβ, but it is key to establishing
connections between different visual pathways.

From a neurophysiological perspective, layer IV-A is formed by stellate
and pyramidal spiny neurons, which have moderately sized somas. These
neurons present a radial dendritic arrangement that facilitates the
integration of visual signals from multiple sources. Their sensitivity
includes spatial details and dynamic properties, allowing this layer to
act as a bridge between the perception of motion (magnocellular pathway)
and color and details (parvocellular pathway). Its synaptic activity is
mainly excitatory, although it also presents inhibitory modulations to
regulate the integration of redundant signals.

In the computational model, this phase focuses on integrating the
signals from IV-Cα and IV-Cβ, consolidating information on motion,
luminance, color, and spatial details. To achieve this, a neural network
will be used to combine the features extracted by both pathways,
generating an output that maintains coherence between the different
aspects of the visual stimulus. In addition, the effect of cortical
feedback from V2 (interleaved bands) will be simulated, which
dynamically adjusts the output of IV-A based on the visual context.

Signal integration is mathematically modeled as a weighted combination
of the inputs from IV-Cα and IV-Cβ:

$$S_{integrado} = \alpha \bullet S_{magno} + \beta \bullet S_{parvo}$$

where 𝑆 ~integrated~ is the consolidated signal, 𝑆 ~magno~ and 𝑆 ~parvo~
are the signals coming from IV-Cα and IV-Cβ, respectively, and 𝛼 and 𝛽
are weights that regulate the contribution of each pathway and are
adjusted during training.

To avoid redundancies in the processed signals, the model applies a
suppression filter that reduces the influence of redundant regions:

$$R(x) = S_{integrado}(x) - \lambda \bullet \sum_{y\epsilon N(x)}^{}{}S_{integrado}(y)$$

where 𝑅 (𝑥 ) is the refined signal at position 𝑥 , 𝑁 ( 𝑥 ) represents
the neighbors of 𝑥 , and 𝜆 is a suppression parameter learned during
training.

Finally, V2\'s recurring feedback is dynamically integrated using:

$$O_{modificado} = O_{original} + \ \gamma \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output of IV-A, 𝐹 ~feedback~ is the
feedback from V2, and 𝛾 is a factor that regulates the influence of the
feedback.

Model validation includes visual tests to analyze how motion, color, and
detail cues are integrated into a coherent representation, functional
tests to assess how V2 feedback modulates IV-A outputs, and performance
tests to measure metrics such as accuracy and loss in signal integration
tasks.

The information flow in this phase begins with input signals processed
by IV-Cα and IV-Cβ, which are integrated and refined in IV-A. The output
of IV-A is then directed to the upper layers of V1 for integration with
other visual pathways and to the intercalated bands of V2, where
advanced visual pattern analysis is performed.

*Phase 12: V1 -- Layers II and III*

Layers II and III of the primary visual cortex (V1) play an essential
role in the integration and distribution of visual information to other
cortical areas. These layers consolidate the signals from layers IV
(IV-Cα, IV-Cβ, IV-A) to prepare their transmission to the secondary
visual cortex (V2) and other associative regions. The integration of
different visual pathways (magnocellular, parvocellular and
koniocellular) that occurs in these layers is key to the contextual and
detailed perception of the visual stimulus.

Neurophysiologically, these layers are predominantly composed of small
to medium-sized pyramidal neurons, whose radial dendritic arrangement
allows them to integrate signals from diverse sources. These neurons are
sensitive to visual properties such as color, orientation, and binocular
disparity, making them critical points for advanced visual pattern
analysis. Synaptic activity in these layers is predominantly excitatory,
but there are also inhibitory interneurons that modulate signals to
avoid redundancies.

The computational model for this phase seeks to emulate these properties
through a neural network that integrates the signals coming from layers
IV of V1. This model is designed to process information about
orientation, color and binocular disparity, consolidating the signals
into a coherent representation. In addition, a feedback mechanism from
V2 is included, which dynamically adjusts the outputs of layers II and
III based on the visual context, simulating cortical interaction.

The integration of signals from IV-Cα, IV-Cβ and IV-A is mathematically
modeled as a weighted combination:

$$I_{II/III} = \ \sum_{k}^{}{}W_{k} \bullet I_{k}$$

where 𝐼 ~II/III~ is the consolidated signal in layers II and III, 𝐼 ~𝑘~
are the signals coming from layers IV, and 𝑤 ~𝑘~ are weights that
regulate the contribution of each layer, adjusted during training.

Processing orientation and binocular disparity is also key in these
layers. Orientation is modeled using oriented convolutional filters,
while binocular disparity is computed as:

$$D(x) = \left| L(x) - R(x) \right|$$

where 𝐷 (𝑥 ) is the disparity at position 𝑥 , and 𝐿 ( 𝑥 ) and 𝑅 ( 𝑥 )
represent the signals from the left and right eyes, respectively.

In addition, recurrent feedback from V2 is incorporated into the model
to dynamically adjust outputs. This is represented by:

$$O_{modificado} = O_{original} + \ \delta \bullet F_{feedback}$$

where 𝑂 modified is the adjusted output, 𝐹 ~feedback~ is the feedback
from V2, and 𝛿 is a parameter that regulates the influence of this
feedback on the final output.

Model validation includes visual tests to assess how integrated signals
generate coherent responses to orientation, color, and binocular
disparity. Functional tests will also be performed to verify the impact
of V2 *feedback* on layer II and layer III outputs, as well as
performance tests to measure metrics such as accuracy and loss on
advanced visual integration tasks.

The information flow in this phase begins with input signals from layers
IV (IV-Cα, IV-Cβ, IV-A). These signals are integrated in layers II and
III using convolutional networks, producing a refined output that is
directed to the upper layers of V1 and to V2, where more advanced
analysis of visual patterns is performed.

*Phase 13: V1 -- Layer V*

Layer V of the primary visual cortex (V1) is a crucial region in the
transmission of visual signals to subcortical structures, such as the
superior colliculus, and in the modulation of visual responses in higher
cortical areas. This layer plays an integrative role, connecting the
initial visual perception in V1 with the areas responsible for
visuomotor integration and eye movement control. The signals processed
in this layer are closely linked to the spatial localization of relevant
visual stimuli, an essential component for orientation and visual
reflexes.

Neurophysiologically, layer V is composed primarily of large pyramidal
neurons, characterized by an extensive dendritic arrangement that
reaches both the upper layers of V1 and subcortical structures. These
neurons are highly sensitive to spatial localization and participate in
the modulation of excitatory signals by inhibitory interneurons. This
allows fine-tuning visual responses before their transmission, ensuring
that only relevant information is processed at higher stages or sent to
subcortical areas.

In the computational model, this phase emulates these functions using a
neural network designed to identify the location of relevant visual
stimuli and transmit this information to subcortical structures and
higher cortical areas. The model integrates signals from layers II and
III of V1, where visual information has already been refined, and
generates an output focused on spatial relevance. Additionally, a
feedback mechanism from higher areas, such as V2 and V3, is
incorporated, which allows the outputs of layer V to be dynamically
adjusted based on the visual context.

Spatial localization is mathematically modeled using a spatial gating
mechanism that prioritizes the relevance of stimuli based on their
immediate environment:

$$L(x) = w_{xy} \bullet S(y)\ $$

where 𝐿 (𝑥 ) represents the spatial response at position 𝑥 , 𝑁 ( 𝑥) is
the neighborhood of 𝑥 , 𝑤 ~𝑥𝑦~ are the weights that regulate the
influence of the neighbors 𝑦 , and 𝑆 ( 𝑦 ) is the input signal at
position 𝑦 . This model ensures that the most relevant signals are
prioritized during processing.

Signal transmission to subcortical structures, such as the superior
colliculus, is organized to maximize the relevance of the information
sent:

$$T_{relevante} = \sum_{x\epsilon R}^{}{}R(x)$$

where 𝑇 ~relevant~ is the total transmitted signal, 𝑅 (𝑥 ) is the
refined signal at position 𝑥 , and 𝑅 represents the relevant positions.

Finally, recurrent feedback from higher areas (V2, V3) is dynamically
integrated into the model by:

$$O_{modificado} = O_{original} + \ \eta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ is the feedback
from higher areas, and 𝜂 regulates the influence of this feedback on the
final output.

Model validation includes visual tests to confirm that the model
correctly identifies the location of relevant visual stimuli, functional
tests to assess the impact of feedback from higher areas (V2, V3) on the
modulation of outputs, and performance tests measuring accuracy and loss
in spatial localization and salience tasks.

The information flow in this phase begins with the input of integrated
signals from layers II and III of V1. These signals are refined in layer
V, where the relevant stimuli are prioritized, and outputs are generated
that are transmitted to the superior colliculus and to the higher areas
of V1 and V2.

*Phase 14: V1 -- Layer VI*

Layer VI of the primary visual cortex (V1) is a key region in visual
processing, as it connects V1 to the Lateral Geniculate Nucleus (LGN)
and closes the thalamocortical loop. This loop is essential for
dynamically adjusting visual perception based on context and visual
attention. Furthermore, layer VI modulates signals transmitted to higher
cortical areas, acting as a filter that refines visual information
before its advanced analysis in associative cortical regions.

From a neurophysiological perspective, layer VI is predominantly
composed of large pyramidal neurons and fusiform neurons, characterized
by their large neuronal somas and dendrites that extend into superficial
layers and subcortical areas. These neurons exhibit mostly excitatory
synaptic activity but are modulated by inhibitory interneurons that
refine the feedback sent to the LGN. This organization allows layer VI
to play an essential role in the integration of descending and ascending
signals, optimizing both the feedback to the LGN and the signals
directed to higher cortical areas.

In the computational model, this phase emulates the thalamocortical
cycle, integrating recurrent signals from higher cortical areas and
adjusting the outputs to the LGN. In addition, the model includes
modulation of ascending cortical signals, allowing the system to
prioritize relevant information based on the visual context. To do so, a
neural network is implemented that processes these interactions,
integrating a feedback mechanism that dynamically adjusts the outputs
based on contextual signals coming from higher areas, such as V2 and V3.

Feedback to the LGN is modeled as a signal modulated by cortical
activity and local inhibition, using the following formula:

$$F_{NGL} = \alpha \bullet I_{cortical} + \beta \bullet R_{inhibitorias}$$

where 𝐹 ~LGN~ is the signal sent to the LGN, 𝐼 ~cortical~ represents the
ascending cortical information, 𝑅 ~inhibitory~ is the local inhibitory
modulation, and 𝛼 and 𝛽 are parameters that regulate the influence of
each component.

On the other hand, the modulation of ascending signals towards higher
areas is modeled by:

$$O_{modificado} = O_{original} + \gamma \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted signal, 𝐹 ~feedback~ is the feedback
from higher areas, and 𝛾 regulates the influence of this feedback on the
final output.

Finally, to ensure that only relevant signals are transmitted to the
NGL, a suppression filter is applied:

$$R(x) = S(x) - \lambda \bullet \sum_{y \in N(x)}^{}{}S(y)$$

where 𝑅 (𝑥 ) is the refined signal, 𝑁 ( 𝑥 ) represents the neighbors of
𝑥 , and 𝜆 is a learned suppression parameter.

Model validation includes visual tests to assess how modulated signals
are sent to the LGN and higher areas, functional tests to analyze the
impact of feedback from higher areas (V2, V3) on the modulation of
signals, and performance tests to measure accuracy and loss in visual
feedback and salience tasks.

The information flow in this phase begins with input signals from the
upper layers of V1 and feedback from higher areas (V2, V3). These
signals are processed in layer VI, where they are refined and
prioritized before being sent to the LGN, closing the thalamocortical
loop, and on to higher areas of V1 and V2 for more advanced analysis.

*Phase 15: V1 -- Layer I*

Layer I of the primary visual cortex (V1) plays a pivotal role in
modulating visual processing by receiving recurrent signals from higher
cortical areas and sending them to deeper layers of V1. Although not
directly involved in the integration of visual features, layer I acts as
a convergence point for contextual cues that dynamically adjust local
visual processing. This makes it an essential component for adapting
visual perception to conditions of attention and context.

Neurophysiologically, layer I is composed primarily of apical dendrites
of pyramidal neurons originating from deeper layers, as well as
inhibitory interneurons that regulate the activity of these dendrites.
This organization allows layer I to receive recurrent signals from
higher areas, such as V2 and V3, and use them to modulate activity in
deeper layers. The integration of these signals ensures that visual
perception in V1 is adjusted to the specific conditions of each visual
context.

In the computational model, layer I is emulated by a system that
integrates recurrent signals from higher areas and local signals from
V1. Modulation is modeled as a weighted combination of recurrent and
local inputs:

$$M_{I} = \alpha \bullet R_{superior} + \beta \bullet L_{local}$$

where 𝑀 ~I~ is the modulated signal in layer I, 𝑅 ~upper~ represents the
recurrent signals from higher areas, 𝐿 ~local~ are the local signals
from V1, and 𝛼 and 𝛽 are parameters that regulate the contribution of
each component.

Deep layer modulation is implemented by dynamically adjusting the
outputs of these layers based on the activity in layer I:

$$O_{modificado} = O_{original} + \gamma \bullet M_{I}$$

where 𝑂 ~modified~ is the adjusted output of the deep layers, 𝑀 ~I~ is
the modulated signal in layer I, and 𝛾 is a parameter that regulates the
influence of this modulation.

Additionally, to ensure that only relevant signals are modulated, a
suppression filter is applied to eliminate redundancies:

$$R(x) = M_{I}(x) - \lambda \bullet \sum_{y \in N(x)}^{}{}M_{I}(y)$$

where 𝑅 (𝑥 ) is the refined signal, 𝑁 ( 𝑥 ) represents the neighbors of
𝑥 , and 𝜆 is a learned parameter that regulates the suppression of
redundant signals.

Model validation includes visual tests to assess how recurrent signals
from higher areas modulate the outputs of deep layers of V1. Functional
tests will also be performed to verify that the modulated signals
dynamically adjust processing based on the visual context. Finally,
performance tests will be implemented to measure accuracy and loss in
visual modulation tasks.

The information flow in this phase begins with the input of recurrent
signals from higher areas, such as V2 and V3, and local signals from V1.
These signals are integrated in layer I to generate modulated outputs
that fine-tune the deep layers of V1.

*Phase 16: V2 -- Thick Bands*

The thick bands of the secondary visual cortex (V2) are essential for
advanced processing of motion and orientation. These bands receive
information directly from layer IV-B of V1, which specializes in
analyzing luminance and motion. In V2, the thick bands refine this
information, identifying directions and speeds of motion and organizing
it for transmission to higher areas such as the medial temporal cortex
(MT) and V3. This processing is critical for dynamically interpreting
the visual environment and for tasks such as spatial navigation and
perception of moving objects.

From a neurophysiological point of view, the thick bands of V2 are
composed mainly of pyramidal neurons with larger receptive fields than
those of V1. This allows them to integrate signals from multiple sources
and perform more detailed analysis of motion and orientation. These
neurons are highly sensitive to directional changes and orientation
gradients, and their synaptic activity is predominantly excitatory,
modulated by inhibitory interneurons that filter redundant signals and
fine-tune responses. This functional structure makes the thick bands a
key point for the dynamic processing of visual information.

The computational model emulates these functions using a deep
convolutional neural network (CNN), specifically designed to analyze
directions, velocities, and gradients of motion in signals from V1. This
model uses oriented convolutional filters that allow the identification
of specific patterns of motion and orientation. In addition, a feedback
mechanism from higher areas, such as MT and V3, is incorporated, which
dynamically adjusts the outputs of the thick bands based on the visual
context.

Motion direction and orientation processing is modeled using specialized
convolutional filters:

$$D_{orientación} = arg\left( \sum_{i}^{}{}w_{i,\theta} \bullet g_{\theta}(x_{i}) \right)\ $$

where 𝐷 ~orientation~ is the estimated direction, 𝜃 represents the
possible orientations, 𝑔 ~𝜃~ (𝑥 ) is a filter oriented towards direction
𝜃 , and 𝑤 ~𝑖\ ,\ 𝜃~ are weights adjusted during training.

To estimate the speed of movement, spatial and temporal changes in the
signals are analyzed:

$$v = \frac{\mathrm{\Delta}x}{\mathrm{\Delta}t}$$

where 𝑣 is the estimated velocity, Δ 𝑥 is the spatial displacement, and
Δ 𝑡 is the time interval between samples. This model ensures that the
thick bands of V2 can accurately identify both the direction and
velocity of motion.

Furthermore, cortical feedback from higher areas (MT and V3) dynamically
adjusts the outputs of the thick bands:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of this feedback.

Model validation includes visual tests to confirm that the network
correctly identifies movement directions and speeds in input signals.
Functional tests will also be performed to analyze how cortical
*feedback* modulates the outputs of the thick bands, and performance
tests to measure accuracy and loss in orientation and movement tasks.

The information flow in this phase begins with the input of IV-B signals
from V1, related to motion and luminance. These signals are refined in
the coarse bands of V2, which process directions, speeds, and
orientation. Finally, the output is directed to higher areas such as MT
and V3, where advanced motion analysis is performed.

*Phase 17: V2 -- Thin Bands*

The thin bands of the secondary visual cortex (V2) play a key role in
analyzing chromatic information and fine spatial patterns. These bands
receive signals primarily from layer IV-Cβ of V1, which processes
high-resolution details and chromatic differences. In V2, the thin bands
refine this information, allowing for more accurate discrimination of
color and complex patterns, and then transmit it to higher areas, such
as V4, where more advanced color and shape analysis is performed.

From a neurophysiological point of view, the thin bands of V2 are
predominantly formed by pyramidal neurons with larger receptive fields
than those of V1. This allows them to combine signals from multiple
chromatic channels and spatial patterns, generating more complete
representations of visual features. These neurons show a high
sensitivity to color gradients and spatial details, with excitatory
synaptic activity that is modulated by inhibitory interneurons, which
refine responses and eliminate redundancies.

The computational model emulates these functions using a convolutional
neural network (CNN) designed to analyze chromatic differences and fine
spatial patterns in the signals coming from V1. The network includes
specialized filters to discriminate complex colors and textures, as well
as a feedback mechanism from higher areas, such as V4, that dynamically
adjusts the outputs of the thin bands based on the visual context.

Chromatic discrimination is modeled by weighted combinations of color
channels:

$$C_{color} = \alpha \bullet R + \beta \bullet G + \gamma \bullet B$$

where 𝐶 ~represents~ the chromatic discrimination, and 𝛼, 𝛽 , 𝛾 are
parameters that adjust the contribution of each chromatic channel. This
formula allows to identify precise differences between hues and to
highlight specific chromatic patterns.

On the other hand, fine spatial pattern analysis is performed using
high-resolution convolutional filters that extract detailed texture and
edge features:

$$P(x) = \sum_{k}^{}{}w_{k} \bullet g_{k}(x)$$

where 𝑃 (𝑥 ) is the representation of the pattern at position 𝑥 , 𝑔 ~𝑘~
( 𝑥 ) are features that activate specific patterns, and 𝑤 ~𝑘~ are
weights tuned during training. This ensures that the network can
accurately identify and categorize complex textures and shapes.

Furthermore, cortical feedback from higher areas, such as V4,
dynamically adjusts the outputs of the thin bands based on the visual
context:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ is the feedback
from V4, and 𝛿 regulates the influence of this feedback on the final
output.

Model validation includes visual tests to verify that the network can
discriminate chromatic differences and fine spatial patterns in the
input signals. Functional tests will also be performed to analyze how V4
feedback modulates the thin band outputs, as well as performance tests
to measure accuracy and loss in color discrimination and pattern
analysis tasks.

The information flow in this phase begins with the input of processed
signals in IV-Cβ of V1, related to color and fine spatial details. These
signals are refined in the thin bands of V2, and the output is directed
to higher areas such as V4, where more advanced analysis of color and
complex shapes is performed.

*Phase 18: V2 -- Interleaved Bands*

The intercalated bands of the secondary visual cortex (V2) play an
essential role in integrating multiple visual features, such as edges,
textures, and color combinations. These bands function as processing
nodes that combine information from the thick and thin bands of V2, as
well as from layers IV-Cα and IV-Cβ of V1. Their main function is to
generate more complete visual representations that are then transmitted
to higher areas, such as V3 and V4, where more advanced analysis is
performed.

From a neurophysiological point of view, intercalated bands are
predominantly formed by pyramidal neurons, characterized by their large
receptive fields that allow them to combine spatial and chromatic
signals. These neurons are highly sensitive to complex textures and
integrated edges, and their synaptic activity is mostly excitatory,
modulated by inhibitory interneurons that refine the signals and
eliminate redundancies. This neurophysiological design allows
intercalated bands to act as an integrating bridge between different
visual pathways.

The computational model for this phase simulates these functions using a
convolutional neural network (CNN) that processes edges, textures, and
color combinations from signals coming from the thick and thin bands.
The network uses a weighted integration system to combine these
features, generating a coherent output that represents the visual
interactions. In addition, the model incorporates a feedback mechanism
from higher areas, such as V3 and V4, to adjust the outputs based on the
visual context.

The integration of signals from the thick and thin bands is modeled by a
weighted sum:

$$I_{intercaladas} = \alpha \bullet I_{gruesas} + \beta \bullet I_{delgadas}$$

where 𝐼 ~interleaved~ is the integrated signal, 𝐼 ~thick~ and 𝐼 ~thin~
represent the signals from the thick and thin bands, respectively, and 𝛼
and 𝛽 are parameters that tune the contribution of each pathway. This
model ensures that the relevant features of motion, orientation, color,
and texture are integrated in a coherent manner.

Complex texture and edge analysis is performed using convolutional
filters that capture detailed spatial patterns:

$$T(x) = \sum_{k}^{}{}w_{k} \bullet g_{k}(x)$$

where 𝑇 (𝑥 ) represents textures at position 𝑥 , 𝑔 ~𝑘~ ( 𝑥 ) are
activation functions for specific patterns, and 𝑤 ~𝑘~ are weights tuned
during training. This approach allows the model to identify complex
textures and embedded edges with high accuracy.

Furthermore, cortical feedback from higher areas, such as V3 and V4,
dynamically adjusts the outputs of the intercalated bands:

$$O_{modificado} = O_{original} + \gamma \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ is the cortical
feedback, and 𝛾 regulates the influence of these recurrent signals.

Model validation includes visual tests to analyze how the interleaved
bands combine edge, texture, and color cues. Functional tests will also
be performed to assess the impact of feedback from higher areas (V3, V4)
on the interleaved bands outputs, and performance tests to measure
accuracy and loss in visual integration tasks.

The information flow in this phase begins with processed signals
entering the thick and thin bands of V2. These signals are integrated in
the intercalated bands, where motion, color, and texture features are
combined. The output is directed toward higher areas such as V3 and V4,
where more advanced analysis is performed.

*Phase 19: V3A -- Binocular Disparity Integration*

The V3A region of the visual cortex plays a central role in integrating
binocular disparity, an essential feature for depth perception. V3A
receives signals from the deep layers of V1, V2, and from the thick,
intercalated bands of V2, which it combines to compute the differences
between the images projected to the left and right eyes. This
computation allows the reconstruction of three-dimensional information
about the environment, facilitating accurate depth perception and
spatial navigation.

Neurophysiologically, V3A is composed of pyramidal and stellate neurons,
characterized by their large receptive fields and specific sensitivity
to binocular disparity. These neurons are organized to combine monocular
signals from both eyes and compute the differences between the projected
images, modulating their activity to highlight these disparities.
Synaptic activity in V3A is predominantly excitatory but is modulated by
inhibitory interneurons that fine-tune the signals to ensure accurate
computation and avoid redundancies.

In the computational model, this phase emulates binocular disparity
integration functions using a convolutional neural network (CNN)
designed to compute and process the differences between the visual
signals from the left and right eyes. This model generates a depth map
from these disparities and adjusts its outputs based on contextual cues
from higher areas, such as MT, through a cortical feedback mechanism.

The calculation of binocular disparity is mathematically modeled as the
difference between the monocular signals from the left and right eyes:

$$D(x) = \left| L(x) - R(x) \right|$$

where 𝐷 (𝑥 ) represents the disparity at position 𝑥 , and 𝐿 ( 𝑥 ) and 𝑅
( 𝑥 ) are the signals from the left and right eyes, respectively. This
calculation makes it possible to highlight key spatial differences
between the two images.

The integration of these disparities to generate a depth map is
performed by a weighted combination of disparities at different
positions:

$$M_{profundidad} = \sum_{k}^{}{}w_{k} \bullet D_{k}$$

where 𝑀 ~depth~ is the resulting depth map, 𝐷 ~𝑘~ are the disparities at
different positions, and 𝑤 ~𝑘~ are weights adjusted during training.
This approach ensures that the model can integrate binocular information
and generate an accurate three-dimensional representation.

Cortical feedback from higher areas, such as MT, is incorporated into
the model to dynamically adjust V3A outputs:

$$O_{modificado} = O_{original} + \gamma \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛾 regulates the influence of these recurrent
signals.

Model validation includes visual tests to verify that the network
generates accurate depth maps from binocular disparities. Functional
tests will also be performed to analyze how cortical feedback from MT
modulates V3A outputs, and performance tests to measure accuracy and
loss in disparity estimation and depth perception tasks.

The information flow in this phase begins with input from monocular
signals processed in V1 and V2, related to the images from the left and
right eyes. These signals are integrated in V3A to calculate disparities
and generate a depth map, which is transmitted to MT for advanced
analysis of motion in three-dimensional space.

*Phase 20: V3A -- Relative Motion Analysis*

The V3A region of the visual cortex is critical for relative motion
analysis, allowing the visual system to interpret the dynamic
relationships between moving objects and their environment. This ability
is essential for tasks such as spatial navigation, collision detection,
and interaction with the environment. V3A receives motion-related
signals from the thick bands of V2 and the MT region, integrating this
information to calculate the relative speeds and directions of motion
between different regions of the visual field.

From a neurophysiological point of view, V3A is composed primarily of
pyramidal neurons with wide receptive fields, organized to process
complex patterns of relative motion. These neurons combine local and
global signals to compute differences in speed and direction between an
object and its background or between multiple moving objects. Synaptic
activity is predominantly excitatory, modulated by inhibitory
interneurons that refine responses and eliminate redundancies, ensuring
accurate and efficient computation.

The computational model for this phase emulates these functions using a
convolutional neural network (CNN) designed to analyze differences in
speed and direction between different regions of the visual field. The
model generates relative motion maps that represent the dynamic
relationships between objects and their environment. It also includes a
feedback mechanism from higher areas, such as MT, that dynamically
adjusts V3A outputs based on the visual context.

The calculation of relative motion is mathematically modeled as the
difference between local and global velocities:

$$V_{relativo}(x) = \left| V_{local}(x) - V_{global} \right|$$

where 𝑉 ~relative~ (𝑥 ) is the relative velocity at position 𝑥 , 𝑉
~local~ ( 𝑥 ) represents the local velocity at 𝑥 , and 𝑉 ~global~ is the
average velocity of the background. This calculation allows to identify
dynamic differences between regions of the visual field.

The integration of these relative velocities to generate a dynamic
motion map is performed by a weighted combination:

$$M_{movimiento} = \sum_{k}^{}{}w_{k} \bullet V_{relativo,k}$$

where 𝑀 ~motion~ is the relative motion map, 𝑉 ~relative,k~ are the
relative velocities at different positions, and 𝑤 ~𝑘~ are weights
adjusted during training. This model ensures that motion relationships
are accurately represented in a dynamic context.

Cortical feedback from MT is incorporated to adjust V3A outputs:

$$O_{modificado} = O_{original} + \gamma \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
MT feedback, and 𝛾 regulates the influence of these recurring signals.

Model validation includes visual tests to verify that accurate relative
motion maps are generated, functional tests to analyze how cortical
feedback from MT modulates V3A outputs, and performance tests to measure
accuracy and loss in relative motion calculation tasks.

The information flow in this phase begins with the input of signals
related to local and global velocity, coming from the thick bands of V2
and MT. These signals are integrated in V3A to calculate relative motion
and generate dynamic maps that represent the relationships between
objects and their environment. Finally, the output is transmitted to MT,
where a more advanced analysis of motion in three-dimensional space is
performed.

*Phase 21: V3 -- Reconstruction of Three-Dimensional Shapes*

The V3 region of the visual cortex is critical for reconstructing
three-dimensional shapes from two-dimensional signals processed in
previous visual areas such as V1, V2, and V3A. This ability is essential
for depth perception and the interpretation of the three-dimensional
structure of objects in the environment. V3 integrates binocular
disparity, edge, and texture information, combining local and global
signals to generate a coherent, three-dimensional representation of
shapes.

From a neurophysiological point of view, V3 is mainly composed of
pyramidal neurons and stellate neurons, characterized by their wide
receptive fields. These neurons integrate complex visual signals coming
from different areas to reconstruct three-dimensional shapes. They are
highly sensitive to binocular disparity, contours and texture
variations, key features for depth perception and three-dimensional
reconstruction. Synaptic activity in V3 is predominantly excitatory,
modulated by inhibitory interneurons that fine-tune the signals to
ensure accurate and relevant representation.

In the computational model, this phase emulates these functions using a
convolutional neural network (CNN) designed to process binocular
disparity, edge, and texture cues. This model generates
three-dimensional maps representing the structure of objects in space
and dynamically adjusts its outputs using feedback from higher areas,
such as V4 and MT, which provide additional context for the
interpretation of three-dimensional shapes.

The computation of three-dimensional shapes is modeled as the weighted
integration of binocular disparity, edge, and texture cues:

$$F_{3D}(x) = \alpha \bullet D(x) + \beta \bullet B(x) + \gamma \bullet T(x)$$

where 𝐹3D (𝑥 ) is the three-dimensional shape at position 𝑥 , 𝐷 ( 𝑥 ), 𝐵
( ~𝑥~ ), and 𝑇 ( 𝑥 ) are the binocular disparity, edge, and texture
cues, respectively, and 𝛼 , 𝛽 , 𝛾 are parameters regulating the
contribution of each component. This formula allows to integrate the key
visual features required for three-dimensional reconstruction.

The integration of these signals to generate a three-dimensional map is
done by combining visual features in different positions:

$$M_{3D} = \sum_{k}^{}{}w_{k} \bullet F_{3D,k}$$

where 𝑀3D is the three-dimensional map, 𝐹3D ~,k~ are the
three-dimensional shapes at different positions, and ~𝑤𝑘\ are~ weights
~tuned~ during training. This approach ensures that the model can
generate accurate three-dimensional representations from two-dimensional
signals.

Furthermore, cortical feedback from higher areas, such as V4 and MT,
adjusts the outputs of V3 dynamically:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals.

Model validation includes visual tests to verify that the model
generates accurate 3D maps from 2D signals. Functional tests will also
be performed to analyze how feedback from higher areas (V4, MT)
modulates V3 outputs, and performance tests to measure accuracy and loss
in 3D reconstruction tasks.

The information flow in this phase begins with input signals related to
binocular disparity, edges, and textures, processed in V1, V2, and V3A.
These signals are integrated in V3 to generate three-dimensional maps
representing the shapes of objects in space. Finally, the output is
transmitted to V4 for advanced shape and color analysis, and to MT for
three-dimensional motion analysis.

*Phase 22: V4α -- Color Processing*

The V4α region of the visual cortex is one of the most specialized areas
for processing color and complex visual patterns. V4α integrates
chromatic signals from the thin bands of V2 and from layer IV-Cβ of V1,
refining them to generate advanced color representations. This
processing allows the identification of hues, gradients, and chromatic
contrasts, which is essential for object recognition tasks in complex
visual environments and for detailed color perception in various
contextual conditions.

From a neurophysiological point of view, V4α is composed mainly of
pyramidal neurons, characterized by their wide receptive fields that
allow them to process specific combinations of chromatic signals. These
neurons are highly sensitive to color gradients, hue relationships, and
complex chromatic patterns. Their synaptic activity is mostly excitatory
but is modulated by inhibitory interneurons that refine the signals and
eliminate redundancies, adjusting chromatic responses according to the
visual context. This neurophysiological design makes V4α a key node for
advanced color perception.

In the computational model, V4α is emulated by a convolutional neural
network (CNN) designed to process multi-channel chromatic signals (such
as RGB) and generate advanced color maps. This model includes
specialized filters that capture chromatic gradients, blends, and
contrasts, dynamically adjusting the outputs through a feedback
mechanism from higher areas. The model focuses on refining and blending
the chromatic signals, providing accurate representations of the
chromatic features of the visual stimulus.

Advanced color processing is modeled by a weighted combination of
signals from the RGB channels:

$$C_{procesado} = \alpha \bullet R + \beta \bullet G + \gamma \bullet B$$

where 𝐶 ~processed~ represents the chromatic processing, 𝑅, 𝐺 , 𝐵 are
the input chromatic channels, and 𝛼 , 𝛽 , 𝛾 are learned parameters that
weight the contribution of each channel. This model allows to
discriminate chromatic combinations and capture precise relationships
between hues.

Chromatic gradient analysis is performed using convolutional filters
that detect transitions between hues and chromatic patterns:

$$G(x) = \sum_{k}^{}{}w_{k} \bullet g_{k}(x)$$

where 𝐺 (𝑥 ) is the chromatic gradient at position 𝑥 , 𝑔 ~𝑘~ ( 𝑥 ) are
activation functions for specific gradients, and 𝑤 ~𝑘~ are weights tuned
during training. This calculation ensures that the model captures smooth
transitions and complex chromatic patterns.

Furthermore, cortical feedback from higher areas dynamically adjusts V4α
outputs:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals. This mechanism ensures that chromatic processing adapts to the
contextual conditions of the visual stimulus.

Model validation includes visual tests to analyze how the model
discriminates color combinations and color gradients. Functional tests
will also be performed to assess the impact of cortical feedback on V4α
outputs, as well as performance tests to measure accuracy and loss in
advanced color processing tasks.

The information flow in this phase begins with the input of chromatic
signals from the thin bands of V2 and layer IV-Cβ of V1. These signals
are processed in V4α to generate advanced maps of chromatic combinations
and gradients, which are transmitted to V4 for further analysis.

*Phase 23: V4α -- Processing of Complex Shapes*

The V4α region of the visual cortex plays an essential role in the
analysis and recognition of complex shapes, such as curves,
intersections, and closed visual structures. This area processes
integrated visual signals from V1, V2, and V3, generating advanced
representations of patterns and structures that allow the identification
of objects and the understanding of their shape in visually complex
contexts. This processing is crucial for the visual recognition of
objects in dynamic environments and for the detailed perception of their
structural features.

From a neurophysiological point of view, V4α is composed mainly of
pyramidal neurons, which present wide receptive fields capable of
combining local and global signals. These neurons are sensitive to the
orientation of curves, intersections and changes in visual structures.
Their synaptic activity is mostly excitatory but is modulated by
inhibitory interneurons that fine-tune responses and eliminate
redundancies. This architecture allows V4α to analyze complex visual
patterns and generate coherent representations of shapes in space.

In the computational model, this phase emulates these functions using a
convolutional neural network (CNN) designed to process advanced visual
signals and analyze complex shapes. The model uses specialized filters
that identify structural features, such as curves and closed patterns,
generating accurate maps of the shapes in the visual field. In addition,
the model includes a cortical feedback mechanism that dynamically
adjusts V4α outputs based on the visual context, ensuring adaptive
interpretation of the signals.

Complex shape processing is modeled as a combination of specific
patterns detected using convolutional filters:

$$F_{formas}(x) = \sum_{k}^{}{}w_{k} \bullet g_{k}(x)$$

where 𝐹 ~shapes~ (𝑥 ) represents the complex shape at position 𝑥 , 𝑔 ~𝑘~
( 𝑥 ) are activation functions for specific patterns, and 𝑤 ~𝑘~ are
weights adjusted during training. This calculation allows the model to
identify structures such as curves and closed shapes with high accuracy.

To generate a structural map of complex shapes, an integration of the
visual features in different positions is performed:

$$M_{formas}(x) = \sum_{k}^{}{}w_{k} \bullet F_{formas,k}$$

where 𝑀 ~shapes~ is the map of complex shapes, 𝐹 ~shapes,\ 𝑘~ are the
shapes processed at different positions, and 𝑤 ~𝑘~ are adjusted weights.
This approach ensures that the model combines local and global cues to
generate complete representations of the visual structures.

Furthermore, cortical feedback from higher areas dynamically adjusts V4α
outputs:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals.

Model validation includes visual tests to verify that the network
correctly identifies complex shapes such as curves and intersections,
functional tests to assess the impact of cortical feedback on V4α
outputs, and performance tests to measure accuracy and loss on complex
shape analysis tasks.

The information flow in this phase begins with input from visual signals
processed in V1, V2, and V3, related to edges, textures, and patterns.
These signals are integrated in V4α to analyze complex shapes and
generate advanced representations of visual structures. Finally, the
output is directed to V4 for further analysis of color and complex
visual patterns, and to associative areas for integration into advanced
visual processing.

*Phase 24: V4β -- Visual Attention Processing*

The V4β region of the visual cortex plays an essential role in
processing visual attention, allowing the visual system to prioritize
relevant stimuli and focus on key information within the environment.
This ability is critical for tasks that require rapidly identifying
important elements while ignoring irrelevant ones. V4β integrates visual
signals from V4α, V1, and V2, as well as contextual signals related to
behavioral goals and the overall visual context. Processing in V4β
dynamically modulates activity in other visual areas, adjusting it
according to the requirements of the task.

From a neurophysiological point of view, V4β is composed mainly of
pyramidal neurons with attention-modulated receptive fields. These
neurons adjust their activity based on the relevance of the stimulus,
highlighting those elements that are important for advanced visual
processing. Their synaptic activity is mostly excitatory, modulated by
inhibitory interneurons that eliminate redundant signals and improve
processing accuracy. This design allows V4β to act as a dynamic filter
that optimizes the resources of the visual system.

In the computational model, this phase emulates the functions of V4β
using a convolutional neural network (CNN) designed to prioritize visual
stimuli based on their relevance. The model combines visual signals from
areas such as V4α with contextual information, generating attention maps
that highlight the most relevant stimuli. It also includes a feedback
mechanism from higher areas, such as the prefrontal cortex (PFC), that
dynamically adjusts outputs based on the visual context and task goals.

Stimulus-relevant modulation is modeled by a weighted combination of the
stimulus\'s visual features and contextual information:

$$R(x) = \alpha \bullet S_{visual}(x) + \beta \bullet C_{contexto}(x)$$

where 𝑅 (𝑥 ) is the relevance of the stimulus at position 𝑥 , 𝑆 ~visual~
( 𝑥 ) represents the visual features of the stimulus, 𝐶 ~context~ ( 𝑥 )
is the contextual information, and 𝛼 , 𝛽 are parameters that regulate
the influence of each component. This calculation ensures that relevant
stimuli are highlighted in the attention map.

The generated attention map is calculated by integrating the relevance
of the stimuli at different positions:

$$M_{atención} = \sum_{k}^{}{}w_{k} \bullet R_{k}$$

where 𝑀 ~attention~ is the visual attention map, 𝑅 ~𝑘~ are the stimulus
salience at different positions, and 𝑤 ~𝑘~ are weights adjusted during
training. This approach allows the model to prioritize relevant stimuli
efficiently and accurately.

Furthermore, cortical feedback from the prefrontal cortex adjusts V4β
outputs dynamically:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals.

Model validation includes visual tests to analyze how attention maps are
generated and how they prioritize relevant stimuli. Functional tests
will also be performed to assess the impact of cortical *feedback* on
V4β outputs, and performance tests to measure accuracy and loss in
visual attention tasks.

The information flow in this phase begins with the input of visual
signals processed in V4α and contextual signals. These signals are
combined in V4β to generate attention maps that prioritize the most
relevant stimuli. Finally, the output is directed to associative areas,
where more advanced analysis and integration of visual information is
performed.

*Phase 25: V5/MT -- Movement Analysis*

The V5/MT (Middle Temporal) region of the visual cortex is fundamental
for the analysis of motion in the visual field, playing a key role in
the perception of directions, speeds and coherence of motion. This area
integrates signals from V1, V2 and V3, processing both local and global
information to generate dynamic representations that allow the visual
system to interpret motion in complex environments. Analysis in V5/MT is
essential for tasks such as spatial navigation, detection of moving
objects and anticipation of trajectories.

Neurophysiologically, V5/MT is composed primarily of large pyramidal
neurons and intermediate neurons, characterized by their wide receptive
fields and specific sensitivity to the direction and speed of movement.
These neurons combine local motion signals to generate a globally
coherent percept, adjusting their activity through inhibitory
interneurons that fine-tune responses and eliminate redundant signals.
This design allows V5/MT to act as an integrating node for advanced
motion analysis.

In the computational model, this phase emulates the functions of V5/MT
using a convolutional neural network (CNN) designed to process dynamic
visual signals and generate maps representing directions and velocities
in different regions of the visual field. The model includes specialized
filters to detect directions and calculate velocities, integrating these
signals into a global motion map. In addition, the model incorporates
cortical feedback from higher areas, such as MST (Medial Superior
Temporal), to dynamically adjust outputs based on visual context.

Motion direction detection is modeled using oriented convolutional
filters:

$$D_{movimiento}(x) = arg\left( \sum_{i}^{}{}w_{i,\theta} \bullet g_{\theta}(x_{i}) \right)\ $$

where 𝐷 ~motion~ (𝑥 ) represents the direction of motion at position 𝑥 ,
𝜃 are the possible orientations of motion, 𝑔 ~𝜃~ ( 𝑥 ) are filters
oriented toward direction 𝜃 , and 𝑤 ~𝑖\ ,\ 𝜃~ are weights adjusted
during training.

The calculation of movement speed is done by analyzing spatial and
temporal changes in visual signals:

$$V(x) = \frac{\mathrm{\Delta}x}{\mathrm{\Delta}t}$$

where 𝑉 (𝑥 ) is the velocity of the movement at position 𝑥 , Δ 𝑥 is the
spatial displacement of the stimulus, and Δ 𝑡 is the time interval
between samples. This calculation ensures that the model can interpret
both the magnitude and direction of the movement.

Global motion integration is modeled by combining local signals to
generate a coherent map:

$$M_{movimiento} = \sum_{k}^{}{}w_{k} \bullet D_{movimiento,k}$$

where 𝑀 ~motion~ is the global motion map, 𝐷 ~motion,\ *k\ *~are the
directions detected at different positions, and 𝑤 ~𝑘~ are weights
adjusted during training. This approach allows the model to integrate
local and global information to generate accurate representations of
motion.

Furthermore, cortical feedback from MST adjusts V5/MT outputs
dynamically:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals.

Model validation includes visual tests to verify how the network
correctly identifies movement directions and velocities, functional
tests to analyze the impact of MST cortical feedback on V5/MT outputs,
and performance tests to measure accuracy and loss in motion analysis
tasks.

The information flow in this phase begins with the input of motion and
luminance-related signals, processed in V1, V2, and V3. These signals
are integrated in V5/MT to calculate directions, velocities, and
coherence of motion, generating dynamic maps that represent motion in
the visual field. Finally, the output is directed to MST, where advanced
global and predictive motion analysis is performed.

*Phase 26: V5/MT -- Binocular Disparity Integration*

The V5/MT (Middle Temporal) region of the visual cortex plays an
essential role in binocular disparity integration, combining monocular
signals from both eyes to calculate depth relationships in motion. This
processing allows the interpretation of the relative positions of
objects in three-dimensional space and their dynamics, facilitating
tasks such as spatial navigation, depth perception and anticipation of
trajectories in the environment.

From a neurophysiological point of view, V5/MT is composed mainly of
large pyramidal neurons and stellate neurons, characterized by their
wide receptive fields and their specific sensitivity to horizontal and
vertical disparities. These neurons integrate monocular signals from the
left and right eyes, adjusting their responses according to the spatial
differences between both images. Their synaptic activity is
predominantly excitatory, but it is modulated by inhibitory interneurons
that filter irrelevant signals and refine the responses, improving the
accuracy in the calculation of binocular disparity.

In the computational model, this phase emulates binocular disparity
integration functions using a convolutional neural network (CNN)
designed to process monocular signals, compute disparities, and generate
dynamic three-dimensional maps. The model combines disparity information
with motion features to represent spatial and dynamic relationships in
the environment. It also includes a cortical feedback mechanism from
higher areas, such as MST (Medial Superior Temporal), that adjusts
outputs based on visual context.

The calculation of binocular disparity is modeled as the difference
between the monocular signals from the left and right eyes:

$$D_{binocular}(x) = \left| L(x) - R(x) \right|$$

where ~binocular~ 𝐷 (𝑥 ) represents the binocular disparity at position
𝑥 , and 𝐿 ( 𝑥 ) and 𝑅 ( 𝑥 ) are the signals coming from the left and
right eyes, respectively. This calculation allows to identify key
spatial differences for depth perception.

Disparity and motion integration combines these signals with dynamic
features to generate a three-dimensional map:

$$M_{3D} = \alpha \bullet D_{binocular} + \beta \bullet V_{movimiento}$$

where 𝑀 ~3D~ is the dynamic three-dimensional map, 𝐷 ~binocular~ is the
binocular disparity, 𝑉 ~motion~ is the speed of movement, and 𝛼, 𝛽 are
parameters that regulate the contribution of each component. This model
ensures that spatial and dynamic relationships are represented in a
coherent manner.

Cortical feedback from higher areas, such as MST, dynamically adjusts
V5/MT outputs:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals.

Model validation includes visual testing to verify that the network
generates accurate dynamic three-dimensional maps, functional testing to
assess the impact of feedback MST cortical input to V5/MT outputs, and
performance tests to measure accuracy and loss in binocular disparity
and motion integration tasks.

The information flow in this phase begins with input from monocular
signals processed in V1 and V2, related to the images from the left and
right eyes. These signals are integrated in V5/MT to calculate
disparities, combine them with motion features, and generate dynamic
three-dimensional maps representing spatial and dynamic relationships.
Finally, the output is transmitted to MST, where advanced
three-dimensional motion and depth analysis is performed.

*Phase 27: V5/MT -- Construction of Dynamic Spatial Maps*

The V5/MT (Middle Temporal) region of the visual cortex plays a crucial
role in constructing dynamic spatial maps, integrating binocular
disparity, motion, and depth information to generate coherent and
dynamic three-dimensional representations of the environment. This
processing allows the visual system to interpret complex spatial and
dynamic relationships, anticipate trajectories, and coordinate
appropriate motor responses, which is essential for real-time
interaction with the environment.

From a neurophysiological point of view, V5/MT is composed of large
pyramidal neurons and intermediate neurons, which have wide receptive
fields and are organized to integrate multiple dynamic visual signals.
These neurons combine local and global information to build spatial maps
that represent motion and depth relationships in three-dimensional
space. Synaptic activity in V5/MT is mostly excitatory, modulated by
inhibitory interneurons that eliminate redundancies and optimize the
precision of the generated dynamic maps.

In the computational model, this phase emulates the functions of V5/MT
using a convolutional neural network (CNN) designed to integrate dynamic
visual signals, such as binocular disparity, speeds, and directions of
movement, generating dynamic three-dimensional maps. The model combines
local and global signals in a continuous time frame, adjusting its
outputs through cortical feedback from higher areas such as MST and
motor areas. This approach allows for building accurate and dynamic
representations of the visual environment.

The integration of dynamic visual signals is modeled by combining
binocular disparity, motion, and depth:

$$M_{espacial}(x) = \alpha \bullet D_{binocular}(x) + \beta \bullet V_{movimiento}(x) + \gamma \bullet P_{profundidad}(x)$$

where 𝑀spatial (𝑥 ) is the dynamic spatial map at position 𝑥 ,
𝐷binocular ( 𝑥 ), 𝑉motion ( 𝑥 ), and 𝑃depth ( 𝑥 )
~are\ the\ binocular\ disparity,\ motion~ , and depth cues,
respectively, and 𝛼 , 𝛽 , ~𝛾~ are parameters ~that~ weight the
~contribution~ of each component.

To generate a continuous dynamic map, visual signals are integrated into
a time frame:

$$T_{mapa}(t) = \sum_{k}^{}{}w_{k} \bullet M_{espacial,k}(t)$$

where 𝑇 ~map~ (𝑡 ) represents the dynamic spatial map at time 𝑡 , 𝑀
~spatial,k~ ( 𝑡 ) are the processed signals at different positions and
times, and 𝑤 ~𝑘~ are weights adjusted during training. This calculation
ensures that the visual representations are spatially and temporally
coherent.

Furthermore, cortical feedback from MST and motor areas dynamically
adjusts V5/MT outputs:

$$O_{modificado} = O_{original} + \delta \bullet F_{feedback}$$

where 𝑂 ~modified~ is the adjusted output, 𝐹 ~feedback~ represents the
cortical feedback, and 𝛿 regulates the influence of these recurrent
signals.

Model validation includes visual tests to analyze how the network
generates accurate and coherent dynamic spatial maps. Functional tests
will also be performed to assess the impact of cortical feedback from
MST and motor areas on V5/MT outputs, and performance tests to measure
accuracy and loss in dynamic map construction tasks.

The information flow in this phase begins with input from visual signals
processed in V1, V2, and V3, related to binocular disparity, motion, and
depth. These signals are integrated in V5/MT to generate dynamic spatial
maps that represent three-dimensional relationships in real time.
Finally, the output is transmitted to MST and motor areas, where
advanced analysis is performed for visuomotor planning and navigation in
dynamic environments.

**FINAL START: RELATIVE PROXIMITY CLASSIFICATION**

The final output of the model should be designed to make decisions based
on the dynamic 3D maps generated in the last phase (V5/MT -- Dynamic
Spatial Map Construction). To do so, a classification layer will be
included that, through supervised learning, identifies whether an object
is "Closer" or "Farther". In technical terms, this layer will be
implemented using a dense architecture with a sigmoid activation, ideal
for binary problems. The associated loss function will be
binary_crossentropy, designed to minimize the discrepancy between the
model predictions and the true labels of the training set. This design
ensures that the model can translate dynamic visual representations into
a clear and accurate decision, aligned with the human perception of
relative proximity.

To ensure the correct functioning of this layer, it is essential to
perform several technical procedures to ensure the coherence of the
signals coming from the previous phases. This includes scaling and
normalizing the intermediate outputs, the efficient integration of the
relevant features, and the implementation of a continuous pipeline
connecting all the phases of the model. These procedures are described
in detail below.

*Signal Scaling and Normalization:* Signals generated in the previous
phases, such as 3D maps, binocular disparity representations, and motion
signals, must be scaled and normalized before being processed in the
decision layer. Scaling adjusts the signals to a uniform range, for
example \[0,1\], while normalization ensures that the signals have a
mean of 0 and a standard deviation of 1, which facilitates model
training and avoids problems related to the dominance of certain
signals.

Mathematically, scaling is done as:

$$S_{escalado} = \frac{S - min(S)}{(S)\  - min(S)}$$

where 𝑆 represents the original signal, and min(𝑆 ) and max ( 𝑆 ) are
the minimum and maximum values of 𝑆 , respectively. The scaled signals
are then normalized using the Z-Score technique:

$$S_{normalizado} = \frac{S_{escalado} - \mu}{\sigma}$$

where 𝜇 and 𝜎 correspond to the mean and standard deviation of 𝑆
~scaled~ .

*Integration of Previous Outputs:* In order for the model to make
informed decisions, the signals processed in the previous phases must be
integrated into a unified vector representing all relevant information.
This is achieved by concatenating the main outputs of the previous
phases: the dynamic three-dimensional map (𝑀3D ), the ~binocular~
disparity representation ( 𝐷binocular ) ~,~ and the motion signals (
𝑉motion ~)~ :

$$I_{final} = \left\lbrack M_{3D},D_{binocular},V_{movimiento} \right\rbrack$$

If this embedded vector turns out to be high dimensional, a dense layer
will be applied to reduce its size and make it more manageable for the
classification layer.

*Continuous Model Pipeline:* To ensure consistency and optimize the flow
of information, a continuous pipeline will be established that connects
all phases of the model, from data input to final output. This pipeline
will include the following stages:

-   *Initial Preprocessing:* Input signals will be scaled and normalized
    to ensure uniform representation.

-   *Intermediate Processing:* The layers corresponding to V1, V2, V3A,
    V5/MT and MST will generate hierarchical visual features such as
    edges, disparity, motion and dynamic three-dimensional maps.

-   *Signal Integration:* Relevant outputs will be combined using
    concatenation and dimensionality reduction operations.

-   *Classification Layer:* A dense unit with sigmoid activation will
    process the integrated information and generate a binary output
    indicating whether an object is "Closer" or "Farther". The output
    will be defined as:

$$P(x) = \sigma(w^{T}x + b)$$

> where 𝜎 is the sigmoid function, 𝑤 represents the weights of the
> classification layer, and 𝑏 is the bias.

**MODEL TRAINING:**

Training the entire system will involve fine-tuning the weights of all
layers of the model to optimize its performance on the classification
task. Initially, a pipeline will be used that freezes the initial stages
of the model and fine-tunes only the upper layers. This will allow
fine-tuning the decision module without altering the weights already
trained in the previous stages. Subsequently, fine-tuning will be
performed *to ensure consistency throughout the entire processing flow.
During training, key metrics such as precision, recall, and loss* will
be monitored , assessing the progress of the model at each stage.

To train the model and tune the model parameters to minimize the loss
function, these steps will be followed:

-   *Model Initialization:* The neural network structure will be
    initialized with synaptic weights using modern techniques such as He
    Initialization to optimize activation propagation in deep networks.
    Layers associated with specific visual areas (e.g. V1, V5/MT) will
    be initialized with distributions simulating biological connections,
    facilitating specialized learning in the first iterations.

-   *Definition of Loss Function:* The main loss function will be the
    Absolute Disparity Difference (ADL), which calculates the absolute
    difference between predicted disparities and true disparities in
    stereo image pairs. Additionally, combined functions including
    penalties for inconsistencies in dynamic 3D maps, such as spatial
    reconstruction loss, will be explored. The final loss function will
    be empirically tuned using cross-validation.

-   *Optimization Algorithm Selection: The main optimization algorithm
    will be Adam, due to its ability to handle large volumes of visual
    data and dynamically adjust the learning parameters. Advanced
    learning* tuning techniques will also be experimented with. *rate,*
    such as ReduceLROnPlateau, to accommodate model convergence
    patterns. Hybrid optimizers, combining Adam and SGD, could be
    employed to improve stability in the later stages of training.

-   *Model Training:* Training will follow a stage-wise scheme, starting
    with freezing the initial layers to efficiently train the upper
    layers. At each iteration (epoch), forward propagation will be
    performed to compute the model predictions, followed by
    backpropagation to adjust the weights based on the computed loss.
    Batch will be incorporated normalization in the dense layers and
    regularization techniques such as dropout to improve the
    generalization of the model. Each iteration will be evaluated on a
    validation set using specific proximity classification metrics (AUC
    and accuracy).

-   *Training Termination:* Training will terminate when a stopping
    criterion based on the convergence of the loss function on the
    validation set is reached. *Early training will be used stopping*
    with a predefined patience threshold to avoid overfitting. If the
    validation loss stops improving over a given number of epochs,
    training will automatically stop.

**MODEL TEST:**

Once trained, the model will be subjected to functional tests in which
it will be exposed to controlled tasks with specific binocular disparity
and motion configurations. In addition, dynamic simulations will be
performed in which objects change relative position in real time. These
tests will allow us to evaluate whether the model can correctly detect
proximity changes and adapt to complex scenarios. Evaluation metrics
will include accuracy, loss, and area under the ROC curve (AUC), which
will allow us to analyze the model\'s performance at different
classification thresholds.

To test the final model, the test set is used, and a detailed analysis
of its performance is performed:

*Making Predictions on the Test Set:* The final trained model will be
used to make predictions on the test set, covering specific binocular
disparity and motion configurations. In addition to static testing,
dynamic simulations will be included where objects change relative
position in real time, evaluating whether the model can adequately
respond to dynamic scenarios.

*Compute Evaluation Metrics:* Specific evaluation metrics will be
computed to measure model performance on both regression and
classification tasks. For regression (binocular disparity prediction),
metrics such as mean square error (MSE), mean absolute error (MAE), and
coefficient of determination (R²) will be employed. For relative
proximity classification (\"Closest\" or \"Farthest\"), precision,
recall, F1-score, and area under the ROC curve (AUC) will be evaluated.
In dynamic simulations, temporal consistency of predictions will be
measured to assess the model\'s ability to adapt to real-time changes.

*Visualizing Results:* The results obtained will be visualized using
tools such as scatter diagrams, histograms and heat maps (Grad -CAM)
that identify the key regions that influence the model predictions. For
dynamic simulations, time error graphs will be included to allow
analyzing the stability and consistency of the predictions over time.

*Detailed Analysis:* A detailed analysis of the model\'s performance
will be performed, identifying patterns, trends and discrepancies in the
results. This will include comparisons between the model\'s predictions
and empirical data from human experiments to validate its cognitive
plausibility. Particular attention will be paid to discrepancies
observed in dynamic and complex scenarios, analyzing whether these are
due to inherent limitations of the model or to inadequacies in the
training data.

*Analysis of the artificial cognitive model:* Conclusions will be drawn
about the neurophysiological plausibility of the connectionist paradigm
in relation to depth perception, evaluating whether the model adequately
reproduces the biological and functional characteristics of the human
visual system. The implications of the findings will be discussed both
for the theoretical understanding of visual perception and for the
development of more general artificial cognitive models. In addition,
the model\'s capacity to process dynamic signals and its potential to
generalize to other visual perception tasks will be analyzed.
