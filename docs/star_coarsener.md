# StarCoarsener plan and behavior

`StarCoarsener(d, m, contract_d=None)` is a deliberately simple first coarsener for bug-testing the core API.

## Fit rule

For each ordered raw-label pair `(P, C)`, count how many vertices with label `P` have at least `d` children with label `C`. Learn the pair when that count is at least `m`.

If `contract_d` is not supplied, it defaults to `d`. It must satisfy:

```python
2 <= contract_d <= d
```

The parameter `d` is therefore the witness threshold used to decide whether a pair is common, while `contract_d` is the smaller transform-time threshold used after the pair has been learned.

## Transform rule

For each learned pair `(P, C)`, transform contracts all matching `C` children under a `P` parent when there are at least `contract_d` such children and the resulting arity is present in the fitted vocabulary.

The token id is:

```python
("star", parent_label, child_label, arity)
```

For example:

```python
("star", "P", "S", 4)
```

means that four children with raw label `"S"` under a raw-label `"P"` parent were contracted.

The vocabulary recipe for this token is:

```python
P = (-1, -1, -1, -1)
L = (("base", "S"), ("base", "S"), ("base", "S"), ("base", "S"))
A = ()
```

The incoming encoded edge from the parent stores the actual attachment map. In the simple raw-star case, this is usually:

```python
attach_map = (0, 0, 0, 0)
```

because all four roots attach to site `0` of the base parent token.

## Closed-vocabulary arities

The fitted vocabulary contains the arities observed during fit for learned pairs and satisfying the `contract_d` threshold. A transform-time group whose arity was not observed during fit is left uncontracted. This keeps the vocabulary fixed after `fit`, matching the BPE-style model contract.
