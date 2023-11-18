# Problem Statement

The evolution of computational linguistics has seen significant strides
in embedding techniques, promising nuanced representations of words in
vector spaces. However, the aggregation of these embeddings,
particularly in the realm of contextual embeddings, has been a point of
contention.

Firstly, the very act of aggregation, whether it be through averaging
mechanisms or clustering methodologies, carries an inherent risk of
information loss. Each instance of a word in textual data is rooted in a
specific context, influenced by myriad factors ranging from syntactic
structures to broader discursive themes. Aggregating embeddings attempts
to distill this richness into a singular or limited set of
representations, often sidelining the less frequent or less predictable
nuances.

Furthermore, aggregation is underpinned by a slew of assumptions. It
presumes a certain homogeneity within the aggregated data points, often
prioritizing dominant senses at the expense of overshadowing emergent or
peripheral meanings. The granularity and intricacy of language, where
words can adopt varied shades of meanings based on contexts,
temporalities, and audiences, challenges the very foundational premise
of aggregation.

The assertion in [@article] is particularly illuminating in this debate.
Kilgariff delineates the difference between 'lumping' and 'splitting'
when defining word senses. The act of lumping seeks to group together
instances based on perceived similarities, often influenced by factors
like frequency and predictability. In contrast, splitting allows for the
recognition of distinctions, emphasizing the heterogeneity of word
usages. Aggregation, by its nature, leans towards the lumping paradigm.
While it offers the advantage of simplification, it does so at the cost
of potentially overlooking the intricate mosaic of semantic landscapes.
A telling example from Kilgarriff's research elucidates this concern. In
[@article], the sense of the word 'handbag' is explored, highlighting an
unconventional interpretation: handbag-as-weapon. This specific sense,
though valid and meaningful in certain contexts, remains absent from
many conventional dictionaries. Such an omission can largely be
attributed to the infrequent deployment of 'handbag' in this particular
sense. This example underscores the pitfalls of aggregation and the
perils of biasing towards frequency---it demonstrates how meaningful,
albeit less common, interpretations can be easily overlooked. The term
**run** exemplifies the complexities inherent in semantic
predictability. Predominantly associated with locomotion, **run**
encompasses an array of interpretations, from operating machinery to
temporal lateness. However, in the realm of journalism, **run** acquires
a specialized nuance, denoting the publication of a story. This specific
interpretation, although contextually significant, might not be
immediately discernible based on the term's more ubiquitous senses.
Models predisposed to dominant patterns could thus misapprehend or
overlook this journalistic context, underscoring the challenges of
predictability. Essentially, a sense's predictability, or lack thereof,
based on prevalent uses does not diminish its validity, highlighting the
intricacies of semantic modeling.

These challenges can be addressed in practical endeavors by establishing
task-dependent foundations for the aggregation process. Depending on the
specific task, the aggregation can be tailored to prioritize certain
senses, underpinning the rationale for such grouping or clustering.
However, this approach also presents drawbacks. Its inherent
subjectivity means that outcomes are not solely dependent on the raw
data (corpus) but are also influenced by editorial philosophies and the
target audience. These external determinants can introduce biases,
potentially distorting the perceived semantic landscape.

## Motivation

In response to the challenges faced by traditional word embedding
aggregation techniques, a graph-based approach offers a promising
alternative. Graphs, by their nature, are inherently flexible and
adaptable. They can be molded to suit specific needs, accommodating a
range of data types and structures. Graphs also offer a more nuanced
representation of data, allowing for the inclusion of multiple
relationships and connections. This is particularly relevant in the
realm of semantic modeling, where words can be associated with a range
of meanings and contexts. Graphs, therefore, facilitate a deeper, more
comprehensive exploration of semantic terrains without resorting to the
lumping that aggregation methods often impose.

In this methodology, individual words are represented as [nodes](../doc/graph/nodes.md) in a
graph, while their corresponding embeddings serve as node features. The
relationships or edges between these nodes are established based on
features of semantic closeness. Graphs also allow for the use of
multiple types of [edges](../doc/graph/edges.md), representing varied semantic associations.
Utilizing a [temporal graph](../doc/graph/temporal_graph.md) forecasting model, the methodology aspires to
predict the dynamism of a target word's semantic relationships across
time, with other words in the corpus. This allows for an effective
capture and forecasting of word meaning transitions and relational
evolutions. To elucidate, let's consider the lexeme **web**.
Historically situated within a biological or fabric-oriented semantic
realm, its associations were predominantly with entities such as
**spider** or **weave**. In the graph representation, the node **web**
during this epoch would exhibit strong edge weights with these terms.
Yet, with the digital revolution and the advent of the internet, **web**
began its semantic transition. If we were to examine its node in a more
recent temporal slice of the graph, we'd anticipate strengthened
connections with nodes representing **browser**, **online**, and
**internet**. The temporal graph forecasting model, by leveraging
historical data and current embeddings, could predict this evolving
topography of edges for the **web** node, flagging its semantic shift.
The graph-centric methodology for detecting semantic shifts, underpinned
by embeddings as node features, embodies a promising departure from
traditional techniques. However, its effective deployment demands a
conscientious navigation of its inherent challenges, ensuring its
robustness in varied linguistic landscapes.

## Project Objectives

The core ambition of this project is to construct a graph-based model
that intricately captures the relationships between words, offering a
solution that surpasses the constraints of traditional word embedding
aggregation techniques. By representing words and their associated
meanings in this interconnected format, the model aims to provide a
deeper perspective on semantics.

Introducing a temporal dimension becomes essential to capture the
fluidity of language, reflecting its evolving nature. This [dynamic
representation](../doc/graph/temporal_graph.md) will enable insights into how word meanings and
associations shift across time, painting a comprehensive picture of
linguistic transitions. Alongside this, effective node selection is
paramount, as the vocabulary size of a corpus can be vast. By including
words that mirror the topological characteristics or are proximal to the
target word in the graph, the model aims to ensure that the graph is
both comprehensive and precise in its semantic representation.
Integrating techniques such as similarity metrics, distributional
semantics, and unsupervised graph clustering, complemented by insights
from domain experts can also enhance this representation.

Forecasting is another crucial aspect. Utilizing temporal graph
forecasting models, the objective is to predict potential shifts in word
relationships and meanings across different chronological spans. This
predictive approach is further enriched by tapping into external
knowledge bases, such as WordNet and ConceptNet, which promise a deeper,
layered understanding of semantic interconnections.

Continuous evaluation forms the backbone of this endeavor. The outcomes
derived from the initial model iterations will be subjected to rigorous
assessment, leading to iterative refinements to enhance accuracy and
maintain semantic fidelity. The graph-centric approach will be
juxtaposed with traditional aggregation methods, allowing for a critical
examination of the strengths, weaknesses, and unique insights each
methodology brings to the fore. Finally, in recognizing the expansive
nature of language, considerations of scalability and efficiency are
paramount. This project is dedicated to ensuring that the developed
model is adaptable to large vocabularies and complex relationships while
optimizing computational demands.

In conclusion, this project embarks on a journey to delve deep into the
intricate tapestry of language, aiming to unravel broader semantic
themes, linguistic trends, and the cultural implications of observed
semantic shifts.
