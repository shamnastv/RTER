# RTER

Re-implementation of the paper Real-Time Emotion Recognition via Attention Gated Hierarchical Memory Network as part of Advanced Deep Leaning Course.
Following modifications are made in architecture.
Utterance Reader : Instead of taking only max of hidden
units, A weighted sum of hidden units is added. The weights
are calculated using projection to a learnable vector followed
by softmax.

Attention GRU To compute the importance of context
the proposed model use dot product of context and query. It
is possible that the importance of a context is not proportional
to the cosine similarity with the query. So a new method is
suggested in which both query and context is transformed to
other dimension using separate learnable parameters and the
dot product is calculated to get weights.

And Added information to distinguish speakers in a conversation
