# ML_SVM_LOGREG_MORE

This repository demonstrates perceptron, logistic regression, and support vector machine classifiers using small tabular datasets. It focuses on explicit gradient-descent implementations and manual feature engineering while leveraging `scikit-learn` for comparison and visualization.

## Purpose & Objectives
- Implement core classification algorithms from scratch.
- Search hyperparameters (learning rate, regularization strength, SVM C/gamma) for optimal accuracy.
- Visualize decision boundaries in 2-D/3-D to understand model behavior.
- Compare custom implementations with `scikit-learn` baselines.

## Architecture & Modules

### `LogisticRegression.py`
Handles a 3-feature SUV purchase dataset. Lines show preprocessing and scaling:
```
df = pd.read_csv('suv_data.csv')
... df['Gender']=df['Gender'].replace({'Male' : 0 , 'Female':1})
...
data=np.concatenate((np.ones((len(data),1)),data),axis=1)
X_train, X_test, y_train, y_test = train_test_split(...)
```
【F:LogisticRegression.py†L70-L90】

The gradient-descent solver with optional L2 regularization is defined as:
```
def Logistic_Regression_via_GD(P,y,lr,lamda = 0):
    w=np.ones(P.shape[1])
    num_iterations=100
    for _ in range(num_iterations):
        gradient=np.zeros(len(w))
        for sample,label in zip(P,y):
            sigmoid_x=sigmoid(w.T @ sample)
            if(label==1):
                gradient+=(label*sample*(1-sigmoid_x))
            else:
                gradient+=(label*sample*sigmoid_x)
        w+=(lr*gradient + 2 * lamda * w)/len(P)
    return w[1:],w[0]
```
【F:LogisticRegression.py†L123-L140】
The script searches for the best learning rate and regularization via grid search and reports accuracy on the held-out set.

### `Perceptron.py`
Uses an exam score dataset. Data normalization and bias insertion occur here:
```
mm_scaler=MinMaxScaler()
orig_data=mm_scaler.fit_transform(orig_data)
data=mm_scaler.fit_transform(data)
data = np.insert(data, 0, 1, axis=1)
data[labels==-1]*=(-1)
```
【F:Perceptron.py†L32-L44】
The perceptron update loop iterates over samples and adjusts the weight vector whenever a classification is wrong:
```
num_iterations=20
for _ in range(num_iterations):
    for sample_index,sample in enumerate(data):
        if(w.T @ sample <0):
            w = w + lr * sample
```
【F:Perceptron.py†L50-L57】
Results are compared with `sklearn.linear_model.Perceptron` to validate correctness.

### `SVM.py`
Downloads or loads a 2‑D dataset and maps it to polynomial features:
```
def phi_func(x):
    return np.concatenate((np.square(x),x),axis=1)
...
new_features=phi_func(X_train)
model = SVC(kernel='linear', C=10)
model.fit(new_features, y_train)
```
【F:SVM.py†L26-L40】
It then plots decision surfaces for linear, polynomial, and RBF kernels and performs a grid search over gamma and C:
```
for gamma in gammas:
    for c in cees:
        model = SVC(kernel='rbf', gamma=gamma, C=c)
        clf = model.fit(X_train,y_train)
        accs.append(clf.score(X_val, y_val))
        if (clf.score(X_val, y_val) > max_acc):
            ...
```
【F:SVM.py†L150-L166】

## Installation & Environment Setup
1. Install Python 3.10+ and system libraries for building matplotlib (`libfreetype6-dev`, `libpng-dev`, `pkg-config`).
2. `pip install -r requirements.txt`
3. Ensure network access if `SVM.py` needs to download `svm_data_2d.npy`.

## Usage
### Run logistic regression
```bash
python LogisticRegression.py  # trains with grid-search and shows a 3‑D plot
```
### Execute SVM experiments
```bash
python SVM.py  # prints the hyperplane equation and opens accuracy plots
```
These scripts print accuracy metrics to stdout and display figures.

## Outputs & Artifacts
- On execution, matplotlib windows visualize datasets and decision boundaries.
- Console logs show learning rates, hyperparameters, and accuracies.
- Datasets (`*.csv`, `*.npy`) reside in the repository root for reproducibility.

## Development & Contribution Workflow
- Format with **black** and lint with **flake8** before committing.
- Add tests under a `tests/` folder and run with `pytest`.
- Submit pull requests that pass CI and include clear descriptions of algorithmic changes.

## Project Status & Roadmap
This is an **alpha**‑stage educational project. Future work includes CLI argument parsing, saving trained models, and broader hyper‑parameter searches.

## License & Attribution
No license file is present. Adopting an [OSI-approved license](https://opensource.org/licenses) such as MIT or Apache‑2.0 is recommended.
