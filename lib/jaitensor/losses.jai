#: Loss functions.

#: Softmax cross-entropy against one integer class per sample.
pub class SparseCategoricalCrossentropy {
    #: Whether the final layer returns raw logits instead of softmax values.
    pub let from_logits: bool

    #: Configure sparse classification from probabilities or raw logits.
    #:
    #: Use a final `Activation.Softmax` with the default. Set `from_logits` and
    #: use a final `Activation.Linear` when callers need direct logits from
    #: `Sequential.predict`.
    pub fn init(self, from_logits: bool = false) { self.from_logits = from_logits }
}
