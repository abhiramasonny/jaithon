# tiny_lm

A character-level language model, trained on a page of text that is in the
source file. Nothing is downloaded, and the whole run takes a few seconds.

```bash
jaithon run examples/tiny_lm/train.jai
```

It prints the loss falling, the score on text it was never shown, and then
writes 260 characters of its own at two temperatures.

## What it is

The previous 16 characters are looked up in a learned embedding table, laid
end to end, and a two-layer head guesses the seventeenth. That is Bengio's
neural language model — it predates transformers by a decade and is enough to
pick up spelling, spacing, and the shape of the sentences it was shown.

It exercises the part of jaitensor no other example touches: `Embedding`, and
the sparse gradient that flows back into a lookup table.

## It memorises, and that is the point

A typical run:

```
783 samples to learn from, 156 held back, 27 distinct characters
trained 40 epochs in 0.3s, loss 3.219 -> 0.006
a model that guessed uniformly would score 3.296
on the text it never saw it scores 6.066
  -- which is far worse than what it trained on: it memorised the page
```

0.006 on the text it trained on is not a good model, it is a lookup table with
extra steps. The held-out slice is what says so: **6.07, which is worse than
guessing uniformly.** A quarter of a million parameters against seven hundred
samples has room to store every one of them, and it does.

This is why the held-out split is in the example rather than left out to make
the number look better. The training loss alone would have read as a triumph.

The generated text shows the same thing from the other side: it comes out in
long verbatim stretches of the corpus, with the joins between them garbled —
it is reciting, not composing. Turning the temperature up makes it wander off
the memorised path sooner and the wandering is nonsense, because there is no
generalisation underneath to fall back on.

Making it generalise is a matter of more text and fewer parameters, in that
order. The corpus here is about a kilobyte; the shape of the problem does not
change until that is a few megabytes.
