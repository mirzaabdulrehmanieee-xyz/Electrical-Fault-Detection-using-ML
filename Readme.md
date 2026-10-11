# README — How to Run Everything

## 1. Open the Project

Open the `Assignment 2` folder in VS Code. Run all commands from the main project directory, where `data`, `src`, `results`, and `app.py` are located.

## 2. Install Dependencies

Install the required Python libraries:

```bash
python3 -m pip install numpy pandas matplotlib scikit-learn joblib streamlit
```

## 3. Run Data Exploration

Generate the exploratory data analysis plots:

```bash
python3 src/eda.py
```

## 4. Train Logistic Regression Models

Train the Logistic Regression models for Set R and Set Q:

```bash
python3 src/train_logistic.py
```

## 5. Train SVM Models

Train and validate the Linear SVM and RBF SVM models:

```bash
python3 src/train_svm.py
```

This command runs both SVM training scripts and saves the selected models in the `results` directory.

## 6. Generate the SVM Validation Plot

```bash
python3 src/plot_svm_results.py
```

## 7. Test All Models

Evaluate the final models on the test dataset:

```bash
python3 src/test_models.py
```

The output includes test accuracy and confusion-matrix values: TN, FP, FN, and TP.

## 8. Run the Front-End Application

Start the Streamlit application:

```bash
python3 -m streamlit run app.py
```

Open the local URL shown in the terminal to access the application.

## 9. View the Results

The generated files are stored in the `results` directory:

- `figures/` — exploratory plots, Logistic Regression cost plots, SVM validation plot, and front-end screenshot.
- `tables/` — Tables 1–5.
- `analysis.md` — model comparison and discussion.
- `best_linear_svm.joblib` — saved Linear SVM model.
- `best_rbf_svm.joblib` — saved RBF SVM model.
- `theta_R.txt` and `theta_Q.txt` — learned Logistic Regression parameters.
