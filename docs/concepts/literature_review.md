---
bibliography: references.bib
---

# Literature Review

The literature review serves as the foundation for the proposed
research, which aims to explore the effectiveness of diachronic
embeddings for semantic shift tracking in English language data.

In recent years, there has been a growing interest in the automatic
detection of semantic change. Word embeddings have been particularly
popular in this field [@survey], and were used in various studies to
track semantic shift in different languages and domains.
Hamilton[@hamilton-etal-2016-diachronic] and Dubossarsky[@dubossarsky-etal-2017-outta]
trained diachronic word embeddings on corpora spanning long periods of
time to track changes in word meanings and analyze new semantic laws.
There has also been a surge in research on short-term semantic changes,
including analysis of Amazon reviews, news articles, and Twitter data
[@hu-etal-2019-diachronic]. While Kutuzov[@armedconf] used diachronic
embeddings to track occurences of events such as armed conflict.

To track semantic shift, a prototypical representation of a word's
meanings is needed [@montanelli2023survey].
Giulianelli[@giulianelli-etal-2020-analysing] offered a distinction between
form-based approaches, where one embedding per word is used to represent
all its possible meanings in a certain time period, and sense-based
approaches, where contextual embeddings are trained to represent the
meaning of a word at each occurence. Hamilton[@hamilton-etal-2016-diachronic]
and Dubossarsky[@dubossarsky-etal-2017-outta] used form-based approaches to train
independent Word2Vec embeddings on corpora organized by time periods.
They recorded the semantic changepoint and shift score of the target
word at different time periods, after aligning their vector spaces.
While Kim Yoon [@kim-etal-2014-temporal] used vector initialisation to initialize
the embeddings for each word using the embeddings of the previous time
period. However, form-based approached have their limitations,
especially when dealing with polysemy or words with multiple meanings,
as they capture and represent words only through their dominant sense.
This paved the way for more advanced models, which embedded words in
their contextual habitats, heralding a new era of semantic modeling.

As for sense-based approaches, their ability in delineating
relationships between words within a sentence ---particularly leveraging
the attention mechanism inherent in the BERT transformer architecture---
has rendered them apt for tracking semantic shifts. Given that such
methodologies yield distinct embeddings for each occurence of a word, an
aggregation phase is implemented. This phase collates these varied
embeddings, particularly when the word is situated in semantically
analogous contexts, culminating in a prototypical representation that
captures the diverse senses of the word. [@hu-etal-2019-diachronic] used
contextual BERT transformer embeddings trained on an annotated
dictionary corpus to model the distribution of the finite set of senses
of target words across time, after taking the average of each word-sense
embeddings as a prototypical representation of the word-sense. While
[@kanjirangat-etal-2020-sst] used clustering and alignment to aggregate
contextual embeddings into seperate identifiable word meanings across
time. Recently, [@periti-etal-2022-done] employed incremental clustering
to gradually cluster the available embeddings from different time steps
and avoid alignment.

To measure the shift in meanings of a word,
[@schlechtweg-etal-2020-semeval] proposed a graded semantic change score
based on the Jensen-Shannon distance between the sense clusters
distributions. [@giulianelli-etal-2020-analysing] used the average
pairwise euclidean distance between the sense clusters instead.
[@periti-etal-2022-done] employed the cosine similarity between the
target word's sense embedding at each time period and the barycenter of
the sense cluster at the most recent time period. Others (Eg.
[@cook-etal-2014-novel]), used the number of embeddings in each cluster
through time as a novelty score, where the maximum ratio is interpreted
as the semantic shift score. For form-based approaches, the cosine
similarity, between the target's embeddings from different time periods
and the most recent embedding, is the most frequently used measure of
semantic shift.

## Limitations of form-based approaches {#limitations-of-form-based-approaches}

Static word embeddings, by their very design, offer a singular
representation for each lexeme. This design inherently biases towards a
word's dominant or most prevalent sense, effectively sidelining its
subordinate or less common senses. The inability to represent multiple
senses is a significant drawback, especially in diachronic linguistic
studies where a lexeme's subordinate sense in one era might evolve to
become its dominant sense in another. When static embeddings are derived
using smaller window sizes, the resulting vectors predominantly capture
immediate lexical surroundings. The cosine similarity between such
embeddings often reveals interchangeability, aligning them closely with
synonyms or collocations. However, a purely synonym-based understanding
of semantics is reductive. Linguistic studies, grounded in comprehensive
semantic frameworks, advocate a more holistic representation of word
meanings. This includes not just synonyms but also antonyms, hyponyms,
and for verbs, associated agents and roles. Relying solely on synonyms,
therefore, results in a myopic semantic perspective, failing to capture
the multifaceted nature of lexemes. Conversely, when static embeddings
employ larger window sizes, the vectors tend to encapsulate broader
contextual relationships rather than the inherent meanings of the words.
As a result, words that frequently appear in similar contexts, but
aren't necessarily synonymous or interchangeable, exhibit high cosine
similarity. Such embeddings, while capturing contextual relationships,
can often obfuscate the true semantic nuances of words. They may not
necessarily provide insights into the core meanings of lexemes but
rather highlight the contexts they're frequently associated with. Static
word embeddings, while pioneering and effective for certain
applications, exhibit inherent limitations when deployed for detecting
semantic shifts. Their singular representations, coupled with the
constraints posed by varying window sizes, often result in either an
overly narrow or overly broad semantic understanding. For a
comprehensive exploration of diachronic semantics, more dynamic and
multifaceted approaches are imperative.

## Limitations of sense-based approaches {#limitations-of-sense-based-approaches}

Contextual embeddings, such as those derived from BERT, inherently
produce distinct representations for each occurrence of a word based on
its specific context. This granularity, while beneficial for tasks
requiring context sensitivity, poses challenges for semantic shift
detection. The need to aggregate these disparate embeddings becomes
paramount to discern and represent the diverse senses a word may adopt.
Research such as that by [@hu-etal-2019-diachronic] has ventured into
average aggregation of these embeddings. While this approach offers a
consolidated perspective, it isn't devoid of limitations:

The approach mandates a supervised training of sense embeddings. As a
consequence, the possible senses are confined to a pre-established,
dictionary-based set. This presents two primary challenges:

\- The method is not conducive to detecting the emergence of novel
senses over time, instead merely quantifying shifts within the
predefined sense distribution.

\- The reliance on a predetermined set of senses demands manually
labeled data, inherently restricting the approach's scalability and
adaptability.

Alternative methods, like the one proposed by
[@kanjirangat-etal-2020-sst], employ clustering for aggregation. This
approach, however, presents its own set of challenges:

\- Cluster alignment is essential to discern consistent word meanings
across successive temporal instances. Alternatively, an incremental
clustering process is necessitated.

\- Algorithms demanding a predefined cluster count ('K') fail to
accommodate the detection of novel senses emerging over time.

\- Clustering, by its nature, can be influenced by biases inherent in
word forms. Addressing this, researchers have proposed clustering
refinement techniques. Some methods involve removing or merging clusters
with minimal members, while others, as suggested by
[@periti-etal-2022-done], discard clusters deemed insufficiently
informative based on their size relative to the entire embeddings set.
However, such refinement techniques, especially in corpora with
imbalanced word meanings, must be applied judiciously. Even smaller
clusters could be pivotal, offering insights into minority or emerging
senses, potentially overlooked when adopting a one-size-fits-all
refinement strategy.

Contextual embeddings, while offering depth and granularity, present
specific challenges when employed for semantic shift detection. The
complexities of aggregation, whether average-based or clustering-based,
alongside inherent limitations of supervised and predefined approaches,
necessitate an adaptable methodology for accurate and comprehensive
semantic analyses.
