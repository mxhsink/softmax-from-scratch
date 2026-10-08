# Softmax Regression from Scratch

A multi-class classifier built in **pure NumPy**: no scikit-learn model, no `.fit()`. It predicts which of **3 iris species** a flower belongs to from its petal length and width.

**Test accuracy: 96.7%** (a random guess would get about 33%).

---

## What the model does

It uses the classic **iris dataset**: 150 flowers, 50 from each of 3 species.

- **Input:** 2 measurements of a flower, its **petal length** and **petal width** (cm).
- **Output:** a **probability for each species**. The highest one is the prediction.

The trained model's real predictions:

| Petal length | Petal width | Setosa | Versicolor | Virginica | Prediction |
|---|---|---|---|---|---|
| 1.4 cm | 0.2 cm | **97%** | 3% | 0% | Setosa |
| 4.5 cm | 1.5 cm | 1% | **73%** | 26% | Versicolor |
| 5.5 cm | 2.0 cm | 0% | 17% | **83%** | Virginica |

Small petals are setosa, medium petals are versicolor, and large petals are virginica. **The model learned those boundaries from the data itself.**

---

## What I built and learned

### One-hot encoding
The model outputs one probability per class, so the labels need the same shape. One-hot encoding turns each class number into a vector with a 1 in that class's position:

```
class 0  →  [1, 0, 0]
class 1  →  [0, 1, 0]
class 2  →  [0, 0, 1]
```

### Softmax
Turns the model's raw scores into probabilities that add up to 1:

```
scores        [1.0, 2.0, 4.0]
probabilities [0.04, 0.11, 0.84]   → predicts class 2
```

### Cross-entropy loss
Measures how much probability the model gave to the **correct** class: `loss = −log(p)`.

```
p = 0.95  →  loss 0.05   (confident and right)
p = 0.50  →  loss 0.69
p = 0.01  →  loss 4.61   (confident and wrong, heavily punished)
```

**What I learned:** loss and accuracy measure different things. Two models can get the same answers right while one has a higher loss, because it was **less confident** in those answers.

### L2 regularization
Adds a penalty for large weights (`½ α Σθ²`), which pulls every weight slightly toward 0 at each step and makes the model less likely to memorise its training data. The bias is not penalised.

**What I learned:** in the gradient the penalty becomes `α θ`, the slope of the penalty, so bigger weights get pulled harder. On this simple dataset accuracy didn't change, but the validation loss went up (0.110 → 0.326). Part of that is the penalty itself, and the rest is the model becoming less confident.

### Early stopping
Watches the validation loss during training and **stops as soon as it stops improving**, before the model starts overfitting.

**Result:** it stopped at **epoch 282 instead of 5,000**, with the same accuracy, so it needed roughly 18× less training.

---

## Results

| Version | Epochs | Validation accuracy | Test accuracy |
|---|---|---|---|
| Plain softmax | 5,001 | 93.3% | 96.7% |
| + L2 regularization (α = 0.01) | 5,001 | 93.3% | 96.7% |
| + L2 + early stopping | **282** | 93.3% | **96.7%** |

---

## Other things done by hand
- **Bias column** of 1s added to the inputs.
- **Train / validation / test split (60 / 20 / 20)** using