import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import json

# Load data
data = pd.read_csv(r'/home/grads/sanchit23/bn/Admission_Predict_Ver1.1.csv')  # Replace with your actual file path

def discretize_and_augment_data(data, gre_bins, toefl_bins, gpa_bins):
    data_disc = data.copy()
    data_disc['GRE Score_cat'] = pd.cut(data_disc['GRE Score'], bins=gre_bins, labels=False)
    data_disc['TOEFL Score_cat'] = pd.cut(data_disc['TOEFL Score'], bins=toefl_bins, labels=False)
    data_disc['CGPA_cat'] = pd.cut(data_disc['CGPA'], bins=gpa_bins, labels=False)
    data_disc['Admit'] = (data_disc['Chance of Admit '] > 0.9).astype(int)
    data_disc['Non_Academic_Evaluation'] = ((data_disc['SOP'] + data_disc['LOR ']) / 2).round().astype(int)
    data_disc['Final_Academics'] = (0.7 * data_disc['CGPA_cat'] + 0.3 * data_disc['Non_Academic_Evaluation']).round().astype(int)
    return data_disc[['GRE Score_cat', 'TOEFL Score_cat', 'CGPA_cat', 'Research', 'SOP', 'LOR ', 'University Rating', 'Non_Academic_Evaluation', 'Final_Academics', 'Admit', 'Chance of Admit ']]

# Define inductive bias model structure
inductive_bias_model = [
    ('University Rating', 'CGPA_cat'),
    ('TOEFL Score_cat', 'Non_Academic_Evaluation'),
    ('GRE Score_cat', 'Non_Academic_Evaluation'),
    ('CGPA_cat', 'Final_Academics'),
    ('Non_Academic_Evaluation', 'Final_Academics'),
    ('Final_Academics', 'Admit'),
    ('Research', 'Admit'),
    ('Non_Academic_Evaluation', 'Admit'),
    ('LOR ', 'Admit'),
    ('SOP', 'Admit')
]

# Define bin configurations
gre_bins_list = [3, 5, 10, 20]
toefl_bins_list = [5, 10, 20]
gpa_bins_list = [10, 5, 3]

results = []

for gre_bins in gre_bins_list:
    for toefl_bins in toefl_bins_list:
        for gpa_bins in gpa_bins_list:
            print(f"Processing: GRE bins = {gre_bins}, TOEFL bins = {toefl_bins}, GPA bins = {gpa_bins}")
            
            data_disc = discretize_and_augment_data(data, gre_bins, toefl_bins, gpa_bins)
            
            print("Class distribution:")
            print(data_disc['Admit'].value_counts(normalize=True))
            print(f"Number of admits (>0.9 chance): {data_disc['Admit'].sum()}")
            print(f"Total number of applicants: {len(data_disc)}")

            # Ensure all variables are treated as categorical
            le = LabelEncoder()
            for col in data_disc.columns:
                if col != 'Chance of Admit ':
                    data_disc[col] = le.fit_transform(data_disc[col])

            # Split data into 80% training and 20% for testing
            train_data, test_data = train_test_split(data_disc.drop('Chance of Admit ', axis=1), test_size=0.2, stratify=data_disc['Admit'], random_state=42)

            # Apply SMOTE to balance the training data
            smote = SMOTE(random_state=42)
            X_train, y_train = train_data.drop('Admit', axis=1), train_data['Admit']
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            model = BayesianNetwork(inductive_bias_model)

            # Train the model
            model.fit(pd.concat([X_train_balanced, y_train_balanced], axis=1), estimator=BayesianEstimator, prior_type="BDeu")

            # Perform inference
            inference = VariableElimination(model)

            # Evaluate the model on clean test data
            X_test = test_data.drop('Admit', axis=1)
            y_true = test_data['Admit']
            y_pred = []

            for _, row in X_test.iterrows():
                evidence = {col: int(row[col]) for col in X_test.columns}
                try:
                    pred = inference.map_query(['Admit'], evidence=evidence)
                    y_pred.append(pred['Admit'])
                except IndexError:
                    y_pred.append(y_true.mode()[0])

            accuracy_clean = accuracy_score(y_true, y_pred)
            f1_clean = f1_score(y_true, y_pred)

            # Store model structure and marginal probabilities
            model_structure = json.dumps(list(model.edges()))
            marginal_probs = {}
            for node in model.nodes():
                marginal = inference.query([node])
                marginal_probs[node] = marginal.values.tolist()

            results.append({
                'GRE bins': gre_bins,
                'TOEFL bins': toefl_bins,
                'GPA bins': gpa_bins,
                'Accuracy (Clean)': accuracy_clean,
                'F1 Score (Clean)': f1_clean,
                'Model': 'Inductive Bias',
                'Model Structure': model_structure,
                'Marginal Probabilities': json.dumps(marginal_probs),
                'Classification Report (Clean)': json.dumps(classification_report(y_true, y_pred, output_dict=True))
            })

            print(f"Accuracy (Clean): {accuracy_clean:.4f}, F1 Score (Clean): {f1_clean:.4f}")
            print("Classification Report (Clean):")
            print(classification_report(y_true, y_pred))
            print("Model Structure:")
            print(model.edges())
            print("\nMarginal Probabilities:")
            for node in model.nodes():
                marginal = inference.query([node])
                print(f"{node}: {marginal.values}")
            print("\n" + "="*50 + "\n")

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Print summary of results
print("\nSummary of Results:")
print(results_df[['GRE bins', 'TOEFL bins', 'GPA bins', 'Accuracy (Clean)', 'F1 Score (Clean)']])

# Calculate and print correlations
correlation_matrix = data_disc.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

