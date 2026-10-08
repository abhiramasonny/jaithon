#: The quick acceptance for this example: ResNet-8 for two epochs, with the
#: full recipe, must reach 60% on the CIFAR-10 test set inside a minute.
#:
#:     jaithon run examples/cifar_resnet/check.jai
#:
#: Exits non-zero when it does not.
from jaitensor import is_available
from std.time import Instant
from .resnet import Recipe, load_cifar, train_resnet

fn main() -> int {
    if not is_available() {
        print("check: needs a GPU")
        return 1
    }
    let started = Instant.now()
    let recipe = Recipe()
    recipe.depth = 8
    recipe.epochs = 2
    print(recipe.describe())
    let train = load_cifar("train")
    let test = load_cifar("test")
    let outcome = train_resnet(recipe, train, test)
    train.free()
    test.free()
    let seconds = started.elapsed().to_seconds()
    print(f"test accuracy {outcome.test_accuracy:.2f}% after {recipe.epochs} epochs; {seconds:.1f}s in all")
    var failed = false
    if outcome.test_accuracy < 60.0 {
        print(f"check FAILED: test accuracy {outcome.test_accuracy:.2f}% is under 60%")
        failed = true
    }
    if seconds >= 60.0 {
        print(f"check FAILED: {seconds:.1f}s is not under 60s")
        failed = true
    }
    if outcome.epoch_loss[1] >= outcome.epoch_loss[0] {
        print("check FAILED: the training loss did not fall")
        failed = true
    }
    if failed { return 1 }
    print("check passed")
    return 0
}
