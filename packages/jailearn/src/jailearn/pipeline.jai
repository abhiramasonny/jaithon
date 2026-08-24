#: Estimators composed out of estimators: in series, over blocks of columns,
#: and side by side.
#:
#: Nothing here is a model or a transform of its own. A `Pipeline` chains
#: whatever it is handed, a `ColumnTransformer` runs a different transformer
#: over each block of columns of one matrix, and a `FeatureUnion` runs several
#: over all of it. All three are `Estimator`s themselves, so they nest, and so
#: a `GridSearchCV` or a `cross_validate` takes one exactly where it would take
#: a bare model.
#:
#: That last part is why the module exists. A scaler fitted once over the whole
#: training set and then cross-validated has already seen the held-out fold's
#: mean; the same scaler as a pipeline step is refit inside each fold, and the
#: score drops to the honest one.
#:
#: The double-underscore convention is the plumbing a search reaches through.
#: `get_params` returns `scaler__with_mean` beside `scaler` itself, `set_params`
#: splits at the first `__` and forwards the rest, and because composites nest
#: the same way, `union__pca__n_components` addresses three levels without any
#: of them knowing how deep it is.
#:
#: No call here walks a sample. Column selection is `base.select_columns` and
#: concatenation is `base.hstack`, one dispatch each over the data; what this
#: module does on the host is bookkeeping over names and column indices, of
#: which there are as many as there are features.
#:
#: ```
#: let pipe = make_pipeline([StandardScaler(), Ridge(alpha: 1.0)])
#: defer { pipe.free() }
#: pipe.set_params(params_of([("ridge__alpha", 10.0)]))
#: pipe.fit(x, y)
#: pipe.score(x, y)
#: ```
from jaitensor import Tensor
from .base import Estimator, EstimatorKind, Params
from .base import check_fitted, check_matrix, check_width, hstack, param_str, select_columns

#: What `get_params` puts between a child's name and one of the child's own
#: parameters, and what `set_params` splits on. scikit-learn's separator, so a
#: grid written against it transfers a line at a time.
pub const NESTED: str = "__"

#: How a `ColumnTransformer` treats the columns no block claimed.
pub const REMAINDER_DROP: str = "drop"

#: The other way: append them unchanged, after every block's output.
pub const REMAINDER_PASSTHROUGH: str = "passthrough"

fn by_name(children: list[tuple[str, Estimator]]) -> dict[str, Estimator] {
    var out: dict[str, Estimator] = {}
    for (name, child) in children { out[name] = child }
    return out
}

#: Insist a composite's children are named the way the flattening needs them:
#: present, distinct, and free of the separator.
#:
#: A step called `a__b` would make `a__b__c` address two different parameters,
#: and which one it reached would depend on the order the steps were searched
#: in. Refusing the name costs one pass at construction.
fn check_names(children: list[tuple[str, Estimator]], what: str) -> void {
    if children.len() == 0 { throw ValueError(f"{what} needs at least one step") }
    var seen: dict[str, bool] = {}
    for (name, child) in children {
        if name.len() == 0 { throw ValueError(f"{what} has a step with no name") }
        if name.find(NESTED) >= 0 {
            throw ValueError(f"{what} step '{name}' holds '{NESTED}', which set_params splits on")
        }
        if seen.contains(name) { throw ValueError(f"{what} has two steps called '{name}'") }
        seen[name] = true
    }
}

#: Every child's parameters under `child__parameter`, and the children
#: themselves under their bare names.
#:
#: The bare entry is what lets a search swap a whole step -- `{"clf": [SVC(),
#: LogisticRegression()]}` is the first grid anybody writes -- and the prefixed
#: ones are how it reaches `clf__c`. A child that is itself a composite has
#: already flattened its own, so the prefixes stack without this knowing that
#: they do.
fn nested_params(children: list[tuple[str, Estimator]], own: Params) -> Params {
    var out: Params = {}
    for (key, value) in own.items() { out[key] = value }
    for (name, child) in children {
        out[name] = child
        for (key, value) in child.get_params().items() { out[f"{name}{NESTED}{key}"] = value }
    }
    return out
}

#: Sort a `set_params` dictionary into the keys the composite owns, one
#: sub-dictionary per child to forward, and the children being replaced whole.
#:
#: Forwarded keys are gathered and delivered in one call per child, because an
#: estimator that validates a combination -- a solver against a penalty -- would
#: reject the halfway state that two calls pass through.
#:
#: An unrecognised key throws, which is the whole reason this is careful. A
#: typo in a grid that quietly set nothing surfaces as a search whose points all
#: scored alike, hours later and nowhere near the typo.
fn route_params(
    children: list[tuple[str, Estimator]],
    params: Params,
    mine: list[str],
    what: str
) -> tuple[Params, dict[str, Params], dict[str, Estimator]] {
    let known = by_name(children)
    var own: Params = {}
    var forwarded: dict[str, Params] = {}
    var replaced: dict[str, Estimator] = {}
    for (key, value) in params.items() {
        let cut = key.find(NESTED)
        if cut >= 0 {
            let head = key[0:cut]
            if not known.contains(head) {
                throw ValueError(f"{what} has no step '{head}' to take '{key}'")
            }
            var sub: Params = forwarded.get(head, {})
            sub[key[cut + NESTED.len():]] = value
            forwarded[head] = sub
            continue
        }
        if known.contains(key) {
            if value is not Estimator {
                throw ValueError(f"{what} step '{key}' takes another estimator, got {value}")
            }
            replaced[key] = value
            continue
        }
        if not mine.contains(key) {
            throw ValueError(f"{what} has no parameter or step called '{key}'")
        }
        own[key] = value
    }
    return (own, forwarded, replaced)
}

#: Concatenate the pieces and free them.
#:
#: Both composites that lay outputs side by side want this, and the freeing is
#: the half a caller forgets: `hstack` copies every part into a new matrix, so
#: the parts are dead the moment it returns.
fn stack_and_free(pieces: list[Tensor], what: str) -> Tensor {
    if pieces.len() == 0 { throw ValueError(f"{what} produced no columns") }
    let out = hstack(pieces)
    for piece in pieces { piece.free() }
    return out
}

#: Estimators run in series: every step but the last transforms, and the last
#: one does whatever it does.
#:
#: A pipeline is an `Estimator`, so `fit`, `predict`, `predict_proba`,
#: `decision_function` and `score` all go through the transforms and arrive at
#: the right end of the chain, and a cross-validation refits the whole of it
#: once per fold.
#:
#: `free` frees the steps: a pipeline owns what it was built from.
pub class Pipeline: Estimator {
    var steps: list[tuple[str, Estimator]]
    var fitted: bool

    #: Build a pipeline from named steps, in the order they run.
    pub fn init(self, steps: list[tuple[str, Estimator]]) {
        check_names(steps, "Pipeline")
        self.steps = steps
        self.fitted = false
    }

    #: What the last step produces, because that is what the pipeline produces.
    pub fn kind(self) -> EstimatorKind { return self.final_estimator().kind() }

    #: The steps by name, in the order they run.
    pub fn named_steps(self) -> dict[str, Estimator] { return by_name(self.steps) }

    #: The last step: the one that predicts, or the last transform.
    pub fn final_estimator(self) -> Estimator {
        let (name, last) = self.steps[self.steps.len() - 1]
        return last
    }

    #: Run `x` through the first `stop` steps, fitting them on the way when
    #: `fitting`, and say whether the answer is ours to free.
    #:
    #: With no steps to run it is `x` itself, which a one-step pipeline's
    #: `predict` hits every time; freeing the caller's design matrix out from
    #: under them would be a memorable bug.
    fn _feed(self, x: Tensor, stop: int, y: Tensor?, fitting: bool) -> tuple[Tensor, bool] {
        var current = x
        var owned = false
        for index in 0..stop {
            let (name, step) = self.steps[index]
            let next = if fitting {
                step.fit_transform(current, y)
            } else {
                step.transform(current)
            }
            if owned { current.free() }
            current = next
            owned = true
        }
        return (current, owned)
    }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        check_matrix(x)
        let last = self.steps.len() - 1
        let (fed, owned) = self._feed(x, last, y, true)
        defer { if owned { fed.free() } }
        let (name, tail) = self.steps[last]
        tail.fit(fed, y)
        self.fitted = true
    }

    #: Fit every step and hand back what the last one transformed.
    #:
    #: Not `fit` then `transform`: that would run the last step's transform
    #: over data it had already transformed inside its own `fit_transform`,
    #: which for a factorisation is the expensive half.
    pub fn fit_transform(self, x: Tensor, y: Tensor? = null) -> Tensor {
        check_matrix(x)
        let last = self.steps.len() - 1
        let (fed, owned) = self._feed(x, last, y, true)
        defer { if owned { fed.free() } }
        let (name, tail) = self.steps[last]
        let out = tail.fit_transform(fed, y)
        self.fitted = true
        return out
    }

    pub fn transform(self, x: Tensor) -> Tensor {
        check_fitted(self.fitted, "Pipeline")
        #: Every step ran, so `owned` is true and the answer is the last step's
        #: rather than a view of the caller's `x`.
        let (out, owned) = self._feed(x, self.steps.len(), null, false)
        return out
    }

    #: Transform through every step but the last, then let `call` ask the last
    #: one for its answer.
    fn _at_end(self, x: Tensor, call: fn(Estimator, Tensor) -> Tensor) -> Tensor {
        check_fitted(self.fitted, "Pipeline")
        let (fed, owned) = self._feed(x, self.steps.len() - 1, null, false)
        defer { if owned { fed.free() } }
        return call(self.final_estimator(), fed)
    }

    pub fn predict(self, x: Tensor) -> Tensor { return self._at_end(x, |m, fed| m.predict(fed)) }

    pub fn predict_proba(self, x: Tensor) -> Tensor {
        return self._at_end(x, |m, fed| m.predict_proba(fed))
    }

    pub fn decision_function(self, x: Tensor) -> Tensor {
        return self._at_end(x, |m, fed| m.decision_function(fed))
    }

    pub fn score(self, x: Tensor, y: Tensor) -> float {
        check_fitted(self.fitted, "Pipeline")
        let (fed, owned) = self._feed(x, self.steps.len() - 1, null, false)
        defer { if owned { fed.free() } }
        return self.final_estimator().score(fed, y)
    }

    pub fn get_params(self) -> Params { return nested_params(self.steps, {}) }

    pub fn set_params(self, params: Params) -> void {
        let (own, forwarded, replaced) = route_params(self.steps, params, [], "Pipeline")
        let known = by_name(self.steps)
        for (name, sub) in forwarded.items() { known[name].set_params(sub) }
        if replaced.len() == 0 { return }
        #: The displaced estimator is not freed. `set_params` is what a search
        #: calls on an unfitted clone, and whoever supplied the replacement is
        #: usually still holding the step it replaced.
        var rebuilt: list[tuple[str, Estimator]] = []
        for (name, step) in self.steps { rebuilt.push((name, replaced.get(name, null) ?? step)) }
        self.steps = rebuilt
    }

    pub fn clone_unfitted(self) -> Estimator {
        var fresh: list[tuple[str, Estimator]] = []
        for (name, step) in self.steps { fresh.push((name, step.clone_unfitted())) }
        return Pipeline(fresh)
    }

    pub fn is_fitted(self) -> bool { return self.fitted }

    #: The first step's, because the first step is what sees the caller's `X`.
    pub fn feature_count(self) -> int {
        let (name, head) = self.steps[0]
        return head.feature_count()
    }

    pub fn free(self) -> void {
        for (name, step) in self.steps { step.free() }
        self.fitted = false
    }

    fn __str__(self) -> str {
        var names: list[str] = []
        for (name, step) in self.steps { names.push(name) }
        return f"Pipeline(steps={names})"
    }
}

#: A different transformer over each block of columns, the outputs side by side.
#:
#: One block per `(name, transformer, columns)` triple. The named columns are
#: gathered with `base.select_columns`, one dispatch and never a host loop
#: building rows; the transformer sees a matrix of only those columns; and
#: `base.hstack` lays the results out in the order the triples were given.
#: Blocks may overlap, and the columns none of them names are dropped or
#: appended unchanged according to `remainder`.
#:
#: ```
#: let ct = ColumnTransformer([
#:     ("scale", StandardScaler(), [0, 1]),
#:     ("bin", KBinsDiscretizer(n_bins: 4), [2]),
#: ], remainder: REMAINDER_PASSTHROUGH)
#: defer { ct.free() }
#: let features = ct.fit_transform(x)
#: ```
pub class ColumnTransformer: Estimator {
    var blocks: list[tuple[str, Estimator, list[int]]]
    var remainder: str
    var leftover: list[int]
    var width: int
    var fitted: bool

    #: Build a transformer over named column blocks.
    pub fn init(
        self,
        blocks: list[tuple[str, Estimator, list[int]]],
        remainder: str = REMAINDER_DROP
    ) {
        check_names(block_pairs(blocks), "ColumnTransformer")
        if remainder != REMAINDER_DROP and remainder != REMAINDER_PASSTHROUGH {
            throw ValueError(
                f"remainder is '{REMAINDER_DROP}' or '{REMAINDER_PASSTHROUGH}', got '{remainder}'"
            )
        }
        for (name, transformer, columns) in blocks {
            if columns.len() == 0 { throw ValueError(f"block '{name}' selects no columns") }
        }
        self.blocks = blocks
        self.remainder = remainder
        self.leftover = []
        self.width = 0
        self.fitted = false
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Transformer }

    #: The blocks' transformers by name.
    pub fn named_transformers(self) -> dict[str, Estimator] {
        return by_name(block_pairs(self.blocks))
    }

    #: The columns no block claimed, ascending.
    #:
    #: One pass over a `[width]` flag list rather than a difference of index
    #: lists: blocks are allowed to overlap, so a difference would have to
    #: deduplicate, and it would cost a scan of every block per column.
    fn _unclaimed(self, width: int) -> list[int] {
        var taken: list[bool] = []
        for index in 0..width { taken.push(false) }
        for (name, transformer, columns) in self.blocks {
            for column in columns {
                if column < 0 or column >= width {
                    throw IndexError(f"block '{name}' names column {column} of {width} features")
                }
                taken[column] = true
            }
        }
        var rest: list[int] = []
        for index in 0..width { if not taken[index] { rest.push(index) } }
        return rest
    }

    fn _prepare(self, x: Tensor) -> void {
        check_matrix(x)
        self.width = x.shape[1]
        self.leftover = self._unclaimed(self.width)
    }

    #: Each block's output, with the passed-through remainder last.
    fn _pieces(self, x: Tensor, y: Tensor?, fitting: bool) -> list[Tensor] {
        var pieces: list[Tensor] = []
        for (name, transformer, columns) in self.blocks {
            let block = select_columns(x, columns)
            let piece = if fitting {
                transformer.fit_transform(block, y)
            } else {
                transformer.transform(block)
            }
            block.free()
            pieces.push(piece)
        }
        if self.remainder == REMAINDER_PASSTHROUGH and self.leftover.len() > 0 {
            pieces.push(select_columns(x, self.leftover))
        }
        return pieces
    }

    #: Fit each block on its own columns. Separate from `fit_transform` so a
    #: `Pipeline`'s last step does not pay for an output nobody reads.
    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        self._prepare(x)
        for (name, transformer, columns) in self.blocks {
            let block = select_columns(x, columns)
            transformer.fit(block, y)
            block.free()
        }
        self.fitted = true
    }

    pub fn fit_transform(self, x: Tensor, y: Tensor? = null) -> Tensor {
        self._prepare(x)
        let out = stack_and_free(self._pieces(x, y, true), "ColumnTransformer")
        self.fitted = true
        return out
    }

    pub fn transform(self, x: Tensor) -> Tensor {
        check_fitted(self.fitted, "ColumnTransformer")
        check_width(x, self.width, "ColumnTransformer")
        return stack_and_free(self._pieces(x, null, false), "ColumnTransformer")
    }

    pub fn get_params(self) -> Params {
        var own: Params = {}
        own["remainder"] = self.remainder
        return nested_params(block_pairs(self.blocks), own)
    }

    pub fn set_params(self, params: Params) -> void {
        let pairs = block_pairs(self.blocks)
        let (own, forwarded, replaced) = route_params(
            pairs,
            params,
            ["remainder"],
            "ColumnTransformer"
        )
        let known = by_name(pairs)
        for (name, sub) in forwarded.items() { known[name].set_params(sub) }
        self.remainder = param_str(own, "remainder", self.remainder)
        if replaced.len() == 0 { return }
        var rebuilt: list[tuple[str, Estimator, list[int]]] = []
        for (name, transformer, columns) in self.blocks {
            rebuilt.push((name, replaced.get(name, null) ?? transformer, columns))
        }
        self.blocks = rebuilt
    }

    pub fn clone_unfitted(self) -> Estimator {
        var fresh: list[tuple[str, Estimator, list[int]]] = []
        for (name, transformer, columns) in self.blocks {
            fresh.push((name, transformer.clone_unfitted(), [column for column in columns]))
        }
        return ColumnTransformer(fresh, remainder: self.remainder)
    }

    pub fn is_fitted(self) -> bool { return self.fitted }

    pub fn feature_count(self) -> int { return self.width }

    pub fn free(self) -> void {
        for (name, transformer, columns) in self.blocks { transformer.free() }
        self.fitted = false
    }

    fn __str__(self) -> str {
        var names: list[str] = []
        for (name, transformer, columns) in self.blocks { names.push(name) }
        return f"ColumnTransformer(blocks={names}, remainder='{self.remainder}')"
    }
}

fn block_pairs(blocks: list[tuple[str, Estimator, list[int]]]) -> list[tuple[str, Estimator]] {
    var pairs: list[tuple[str, Estimator]] = []
    for (name, transformer, columns) in blocks { pairs.push((name, transformer)) }
    return pairs
}

#: Several transformers over the same columns, their outputs side by side.
#:
#: The parallel to `ColumnTransformer`'s blocks: every part sees all of `X`, and
#: `hstack` puts what they produce next to each other in the order given. A
#: `PCA` beside a `Nystroem` over the same features is what it is for.
pub class FeatureUnion: Estimator {
    var parts: list[tuple[str, Estimator]]
    var width: int
    var fitted: bool

    #: Build a union from named transformers.
    pub fn init(self, parts: list[tuple[str, Estimator]]) {
        check_names(parts, "FeatureUnion")
        self.parts = parts
        self.width = 0
        self.fitted = false
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Transformer }

    #: The parts by name.
    pub fn named_transformers(self) -> dict[str, Estimator] { return by_name(self.parts) }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        check_matrix(x)
        for (name, part) in self.parts { part.fit(x, y) }
        self.width = x.shape[1]
        self.fitted = true
    }

    pub fn fit_transform(self, x: Tensor, y: Tensor? = null) -> Tensor {
        check_matrix(x)
        var pieces: list[Tensor] = []
        for (name, part) in self.parts { pieces.push(part.fit_transform(x, y)) }
        self.width = x.shape[1]
        self.fitted = true
        return stack_and_free(pieces, "FeatureUnion")
    }

    pub fn transform(self, x: Tensor) -> Tensor {
        check_fitted(self.fitted, "FeatureUnion")
        check_width(x, self.width, "FeatureUnion")
        var pieces: list[Tensor] = []
        for (name, part) in self.parts { pieces.push(part.transform(x)) }
        return stack_and_free(pieces, "FeatureUnion")
    }

    pub fn get_params(self) -> Params { return nested_params(self.parts, {}) }

    pub fn set_params(self, params: Params) -> void {
        let (own, forwarded, replaced) = route_params(self.parts, params, [], "FeatureUnion")
        let known = by_name(self.parts)
        for (name, sub) in forwarded.items() { known[name].set_params(sub) }
        if replaced.len() == 0 { return }
        var rebuilt: list[tuple[str, Estimator]] = []
        for (name, part) in self.parts { rebuilt.push((name, replaced.get(name, null) ?? part)) }
        self.parts = rebuilt
    }

    pub fn clone_unfitted(self) -> Estimator {
        var fresh: list[tuple[str, Estimator]] = []
        for (name, part) in self.parts { fresh.push((name, part.clone_unfitted())) }
        return FeatureUnion(fresh)
    }

    pub fn is_fitted(self) -> bool { return self.fitted }

    pub fn feature_count(self) -> int { return self.width }

    pub fn free(self) -> void {
        for (name, part) in self.parts { part.free() }
        self.fitted = false
    }

    fn __str__(self) -> str {
        var names: list[str] = []
        for (name, part) in self.parts { names.push(name) }
        return f"FeatureUnion(parts={names})"
    }
}

#: Names for unnamed estimators: the class name lowercased, and `-1`, `-2`
#: suffixes when the same class appears more than once.
#:
#: Two passes because the suffix depends on the total: one of a kind keeps the
#: plain name, so `make_pipeline([StandardScaler(), Ridge()])` gives
#: `ridge__alpha` and not `ridge-1__alpha`.
fn auto_names(estimators: list[Estimator]) -> list[str] {
    var totals: dict[str, int] = {}
    for estimator in estimators { totals[type_of(estimator).lower()] = 0 }
    for estimator in estimators {
        let name = type_of(estimator).lower()
        totals[name] = totals[name] + 1
    }
    var used: dict[str, int] = {}
    var names: list[str] = []
    for estimator in estimators {
        let name = type_of(estimator).lower()
        if totals[name] == 1 {
            names.push(name)
            continue
        }
        let nth = used.get(name, 0) + 1
        used[name] = nth
        names.push(f"{name}-{nth}")
    }
    return names
}

#: A pipeline whose steps are named after their classes.
pub fn make_pipeline(estimators: list[Estimator]) -> Pipeline {
    let names = auto_names(estimators)
    var steps: list[tuple[str, Estimator]] = []
    for (index, estimator) in estimators.enumerate() { steps.push((names[index], estimator)) }
    return Pipeline(steps)
}

#: A column transformer whose blocks are named after their transformers'
#: classes, from `(transformer, columns)` pairs.
pub fn make_column_transformer(
    blocks: list[tuple[Estimator, list[int]]],
    remainder: str = REMAINDER_DROP
) -> ColumnTransformer {
    var transformers: list[Estimator] = []
    for (transformer, columns) in blocks { transformers.push(transformer) }
    let names = auto_names(transformers)
    var triples: list[tuple[str, Estimator, list[int]]] = []
    for (index, block) in blocks.enumerate() {
        let (transformer, columns) = block
        triples.push((names[index], transformer, columns))
    }
    return ColumnTransformer(triples, remainder: remainder)
}
