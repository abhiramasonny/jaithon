#: Learn to write, one character at a time, from a page of text in this file.
#:
#:     jaithon run examples/tiny_lm/train.jai
#:
#: A language model small enough to train in a few seconds and old enough to
#: predate transformers: the previous `CONTEXT` characters are looked up in an
#: embedding table, laid end to end, and a two-layer head guesses the next
#: one. That is Bengio's neural language model, and it is enough to pick up
#: spelling, spacing and the shape of the sentences it was shown.
#:
#: The corpus is in this file, so nothing is downloaded. It is small on
#: purpose -- the interesting thing about a model this size is not that it
#: generalises, it is watching it go from noise to words to phrases over a
#: handful of epochs, which a page of text is enough to show.
import std.math
from jaitensor import Activation, Adam, Dataset, Dense, Embedding, Sequential
from jaitensor import SparseCategoricalCrossentropy, Tensor, is_available
from std.random import Random
from std.time import Instant

#: How many characters the model sees before guessing the next one.
let CONTEXT = 16
#: Width of each character's learned vector.
let WIDTH = 24
let HIDDEN = 256
let EPOCHS = 40
let BATCH = 128

let CORPUS = "the device does the work and the host says what the work is. " +
    "a kernel is a small program the device runs once for every element. " +
    "the host queues the kernels and waits only when it needs an answer. " +
    "a buffer holds the numbers the kernels read and the numbers they write. " +
    "the tensor knows its shape and the buffer knows its size. " +
    "a graph is a plan the device compiles once and runs many times. " +
    "the compiler folds the small operations into the large ones. " +
    "a gradient flows backward through every operation that made it. " +
    "the optimizer reads the gradient and moves the parameter a little. " +
    "training is that loop, run until the loss stops falling. " +
    "the loss says how wrong the answer was and nothing about why. " +
    "a batch is many samples at once because the device likes wide work. " +
    "the shape of the work decides which kernel is the fast one. " +
    "measure before changing anything and measure again after. " +
    "a number that was not measured is a guess wearing a decimal point. "

#: The characters that occur, in a fixed order, so a run is repeatable.
fn alphabet(text: str) -> list[str] {
    var seen: list[str] = []
    for ch in text {
        if not seen.contains(ch) { seen.push(ch) }
    }
    return seen.sort()
}

fn main() -> int {
    if not is_available() {
        print("this example needs a GPU")
        return 0
    }
    let letters = alphabet(CORPUS)
    let vocab = letters.len()
    var code: dict[str, int] = {}
    for (index, ch) in letters.enumerate() { code[ch] = index }

    var chars: list[str] = []
    for ch in CORPUS { chars.push(ch) }
    let total = chars.len() - CONTEXT

    #: Every position in the text is one sample: the CONTEXT characters
    #: before it, and the character itself as the answer.
    var contexts: list[float] = []
    var answers: list[float] = []
    for at in 0..total {
        for step in 0..CONTEXT { contexts.push(float(code[chars[at + step]])) }
        answers.push(float(code[chars[at + CONTEXT]]))
    }
    #: A slice at the end that the model never trains on. A page of text and
    #: a model this size will memorise rather than generalise, and the only
    #: way to know that is to keep some of it back and ask.
    let held = total // 6
    let learned = total - held
    var train_x: list[float] = []
    var train_y: list[float] = []
    var test_x: list[float] = []
    var test_y: list[float] = []
    for at in 0..total {
        for step in 0..CONTEXT {
            let value = contexts[at * CONTEXT + step]
            if at < learned { train_x.push(value) } else { test_x.push(value) }
        }
        if at < learned { train_y.push(answers[at]) } else { test_y.push(answers[at]) }
    }
    let data = Dataset(Tensor([learned, CONTEXT], train_x), Tensor([learned], train_y))
    let holdout = Dataset(Tensor([held, CONTEXT], test_x), Tensor([held], test_y))
    print(f"{learned} samples to learn from, {held} held back, {vocab} distinct characters")

    let model = Sequential(
        [
            Embedding(vocab, WIDTH),
            Dense(HIDDEN, activation: Activation.Relu),
            Dense(vocab, activation: Activation.Softmax),
        ]
    )
    model.compile(optimizer: Adam(learning_rate: 0.004), loss: SparseCategoricalCrossentropy())
    let started = Instant.now()
    let history = model.fit(
        data,
        epochs: EPOCHS,
        batch_size: BATCH,
        verbose: false,
        plot_path: null
    )
    let seconds = started.elapsed().to_seconds()
    let first = history.loss[0]
    let last = history.loss[history.loss.len() - 1]
    print(f"trained {EPOCHS} epochs in {seconds:.1f}s, loss {first:.3f} -> {last:.3f}")
    #: A uniform guess over `vocab` characters scores `ln(vocab)`, so that is
    #: the line the model has to get under to have learned anything at all.
    let blind = math.log(float(vocab))
    let unseen = model.evaluate(holdout)
    print(f"a model that guessed uniformly would score {blind:.3f}")
    print(f"on the text it never saw it scores {unseen.loss:.3f}")
    if unseen.loss > last * 4.0 {
        print("  -- which is far worse than what it trained on: it memorised the page")
    }

    print("")
    for temperature in [0.4, 0.8] {
        print(f"--- writing at temperature {temperature} ---")
        print(f"  {write_with(model, code, letters, temperature)}")
    }
    return 0
}

#: Generate 260 characters, feeding each guess back in as context.
#:
#: `temperature` divides the scores before they are turned into a choice: low
#: keeps to the likeliest character and repeats itself, high wanders. Sampling
#: rather than always taking the best is what stops it looping on one phrase.
fn write_with(
    model: Sequential,
    code: dict[str, int],
    letters: list[str],
    temperature: float
) -> str {
    let rng = Random.with_seed(11)
    var window: list[float] = []
    let opening = "the device "
    var out = opening
    for ch in opening { window.push(float(code[ch])) }
    while window.len() < CONTEXT { window.insert(0, float(code[" "])) }
    while window.len() > CONTEXT { window.remove_at(0) }

    for _step in 0..260 {
        let scores = model.predict(Tensor([1, CONTEXT], window))
        let row = scores.download()
        #: The head already gives probabilities, so temperature reshapes them
        #: rather than the raw scores: a power below one sharpens, above one
        #: flattens.
        var weights: list[float] = []
        var mass = 0.0
        for value in row {
            let shaped = max(value, 1.0e-9) ** (1.0 / temperature)
            weights.push(shaped)
            mass += shaped
        }
        var pick = rng.next_float() * mass
        var chosen = weights.len() - 1
        for (index, weight) in weights.enumerate() {
            pick -= weight
            if pick <= 0.0 {
                chosen = index
                break
            }
        }
        out += letters[chosen]
        window.push(float(chosen))
        window.remove_at(0)
        scores.free()
    }
    return out
}
