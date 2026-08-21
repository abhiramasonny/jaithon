# digit_trainer

jaicv draws the training set, jaitensor learns it, jaicv draws the results.
Both halves are in this repository and nothing is downloaded — there is no
MNIST here.

```bash
jaithon run examples/digit_trainer/train.jai
```

About nine seconds end to end: four and a half rendering 4,400 digits, three
training 120 epochs. Reaches **99.2% on the held-out set**, and writes
`digit_mistakes.png` — a contact sheet of what it got wrong, with the guess in
red and the truth in green.

## Why generated data works here

Every sample is distorted differently: font face, scale, stroke width, a shift
of a couple of pixels, a rotation up to ±17°, and per-pixel noise, all from a
seeded generator. A network trained on ten pristine glyphs would memorise ten
bitmaps; the distortion is what makes the problem real. The rotation is the
important one — a leaning `1` and a leaning `7` are the easiest pair in the set
to confuse, and the mistakes sheet usually shows exactly that.

The model is a plain three-layer MLP on purpose. A convolutional net would
score higher and make the example about architecture rather than about the two
libraries meeting.
