# Research Methodology

The methodology devised for this research can be categorized into
several phases, each building upon the previous to facilitate a
comprehensive exploration of semantic shifts.

Initially, for every period-specific corpus, we generate word
embeddings. Employing neural network models facilitates the translation
of textual data into high-dimensional vector spaces, ensuring that words
are adequately represented in relation to their semantic and contextual
attributes. This process yields a set of embeddings for each period,
effectively capturing the diachronic nature of language.

Post embedding generation, the subsequent task is to determine which
words merit representation as nodes in the graph for every period. This
decision is informed by a mix of criteria: the embeddings' inherent
semantics, the frequency and distribution of words in the corpora, and
potential significance in tracking semantic shifts. The aim is to ensure
the graph's nodes are both representative of the period and pertinent to
the overarching research objectives. One foundational strategy for node
inclusion in graph-based semantic modeling is determining similarity
scores, like cosine similarity, between the target word and the entire
vocabulary. Words that exceed a set similarity threshold can be
integrated into the graph. Additionally, the principle of distributional
semantics suggests that words sharing contextual proximity often possess
semantic congruence. Thus, including words that frequently co-occur
within a specific window size of the target word enriches the semantic
landscape. Furthermore, unsupervised graph clustering algorithms applied
on word embeddings can delineate clusters of semantically related words,
guiding node selection. Knowledge bases, such as WordNet or ConceptNet,
offer repositories of predefined semantic relationships, allowing for
the inclusion of semantically linked words to the target. Meanwhile,
metrics like Point Mutual Information (PMI) can gauge the strength of
word associations, emphasizing words that share strong contextual
intersections with the target. Techniques such as topological data
analysis emphasize structural semantics, while diversity sampling
ensures a panoramic view of the semantic space. An iterative refinement
based on initial outcomes, coupled with domain expert insights, further
optimizes the graph's semantic representation.

Edges in our graph are designed to represent the relationships between
words (nodes). For each period, the edge features are constructed based
on the cosine similarity or point-mutual-information between word
embeddings, providing a metric of semantic closeness. Additionally, we
categorize these relationships into various edge types, representing
relationships such as synonymy, collocation, context words, or other
semantic associations. Edge weights are then assigned, quantifying the
strength of these relationships.

With the graph structured, we proceed to model the temporal graph data.
For this, we adopt a Dynamic Graph Neural Network infused with temporal
signals, utilizing the PyTorch Geometric library. This state-of-the-art
tool allows for the intricate handling of time-evolving graph data,
accommodating the diachronic nuances of our dataset.

Our inferential approach for semantic shifts spans several
methodologies. Primarily, we predict future edge feature values between
nodes, drawing parallels from methods employed in traffic forecasting
tasks. This enables a projection into potential future linguistic
trends, capturing shifts in word sense usage. Concurrently, we deploy
time series anomaly detection, ensuring that any sudden or unprecedented
semantic shifts are promptly flagged. Link prediction in graphs
supplements this by highlighting how semantic relationships between
words may evolve or transform over time. Lastly, for those periods where
a threshold criteria is applied to streamline the graph, analyzing nodes
that either emerge or fade becomes crucial. This offers insights into
the entrance or exit of particular word meanings in the lexicon.

In addition to the above methods, it might be beneficial to integrate
community detection within the graph. This could identify clusters of
words that share semantic trajectories, potentially unearthing broader
linguistic themes or trends.

In summary, this rigorous methodology, characterized by its depth and
breadth, aspires to meticulously chart the intricate dynamics of
semantic shifts across time, offering invaluable insights into the
evolution of language and meaning.
