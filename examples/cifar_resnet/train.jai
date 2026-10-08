#: ResNet-20 on CIFAR-10, trained from scratch on the GPU.
#:
#:     jaithon run examples/cifar_resnet/train.jai [--epochs 30] [--seed 0]
#:         [--depth 20] [--width 16] [--batch 128] [--lr 0.2] [--pct 0.25]
#:         [--wd 5e-4] [--normalize 1] [--train N]
#:
#: Every epoch draws a fresh random crop (pad 4) and flip for all 50,000
#: images on the GPU, then makes one shuffled pass in batches of 128 with SGD
#: (Nesterov momentum 0.9, weight decay 5e-4) on a one-cycle learning rate.
#: The peer is `cifar_resnet.py`, the same recipe in PyTorch on MPS; the README
#: has both sides' numbers and the commands that measured them.
from jaitensor import is_available
from std.os import argv
from .resnet import Recipe, load_cifar, train_resnet

fn parse(args: list[str]) -> tuple[Recipe, int] {
    let recipe = Recipe()
    var samples = 0
    var index = 1
    while index < args.len() {
        let flag = args[index]
        if index + 1 >= args.len() { throw ValueError(f"{flag} needs a value") }
        let value = args[index + 1]
        if flag == "--epochs" {
            recipe.epochs = int(value)
        } elif flag == "--seed" {
            recipe.seed = int(value)
        } elif flag == "--depth" {
            recipe.depth = int(value)
        } elif flag == "--width" {
            recipe.width = int(value)
        } elif flag == "--batch" {
            recipe.batch = int(value)
        } elif flag == "--lr" {
            recipe.max_lr = float(value)
        } elif flag == "--pct" {
            recipe.pct_start = float(value)
        } elif flag == "--wd" {
            recipe.weight_decay = float(value)
        } elif flag == "--normalize" {
            recipe.normalize = value != "0"
        } elif flag == "--train" {
            samples = int(value)
        } else {
            throw ValueError(f"unrecognised argument {flag}")
        }
        index += 2
    }
    return (recipe, samples)
}

fn main() -> int {
    if not is_available() {
        print("needs a GPU")
        return 1
    }
    var parsed = (Recipe(), 0)
    try {
        parsed = parse(argv())
    } catch failure {
        print(failure.message)
        print("usage: jaithon run examples/cifar_resnet/train.jai [--epochs N] [--seed N] [--depth 6n+2] [--width N] [--batch N] [--lr F] [--pct F] [--wd F] [--normalize 0|1] [--train N]")
        return 2
    }
    let (recipe, samples) = parsed
    print(recipe.describe())
    let train = load_cifar("train", samples, normalize: recipe.normalize)
    let test = load_cifar("test", normalize: recipe.normalize)
    let outcome = train_resnet(recipe, train, test)
    print(f"parameters {outcome.parameters}")
    print(f"train {outcome.train_seconds:.2f}s  steady epoch {outcome.steady_epoch():.3f}s  first epoch {outcome.epoch_seconds[0]:.3f}s")
    print(f"test accuracy {outcome.test_accuracy:.2f}%  loss {outcome.test_loss:.4f}  (batch statistics, batch 1000)")
    train.free()
    test.free()
    return 0
}
