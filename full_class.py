import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import json

# Load data
data = pd.read_csv(r'/home/grads/sanchit23/bn/Admission_Predict_Ver1.1.csv')  # Replace with your actual file path
def discretize_data(data, gre_bins, toefl_bins, gpa_bins):
    data_disc = data.copy()
    data_disc['GRE Score_cat'] = pd.cut(data_disc['GRE Score'], bins=gre_bins, labels=False)
    data_disc['TOEFL Score_cat'] = pd.cut(data_disc['TOEFL Score'], bins=toefl_bins, labels=False)
    data_disc['CGPA_cat'] = pd.cut(data_disc['CGPA'], bins=gpa_bins, labels=False)
    data_disc['Admit'] = (data_disc['Chance of Admit '] > 0.9).astype(int)
    return data_disc[['GRE Score_cat', 'TOEFL Score_cat', 'CGPA_cat', 'Research', 'SOP', 'LOR ', 'University Rating', 'Admit']]

def create_fully_connected_model(nodes):
    return [(node, 'Admit') for node in nodes if node != 'Admit']

# Define bin configurations
gre_bins_list = [3, 5, 10, 20]
toefl_bins_list = [5, 10, 20]
gpa_bins_list = [10, 5, 3]

results = []

for gre_bins in gre_bins_list:
    for toefl_bins in toefl_bins_list:
        for gpa_bins in gpa_bins_list:
            print(f"Processing: GRE bins = {gre_bins}, TOEFL bins = {toefl_bins}, GPA bins = {gpa_bins}")
            
            data_disc = discretize_data(data, gre_bins, toefl_bins, gpa_bins)
            
            print("Original class distribution:")
            print(data_disc['Admit'].value_counts(normalize=True))
            print(f"Number of admits (>0.9 chance): {data_disc['Admit'].sum()}")
            print(f"Total number of applicants: {len(data_disc)}")

            # Ensure all variables are treated as categorical
            le = LabelEncoder()
            for col in data_disc.columns:
                data_disc[col] = le.fit_transform(data_disc[col])

            # Split the data
            X = data_disc.drop('Admit', axis=1)
            y = data_disc['Admit']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Apply SMOTE to the training data
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            print("Resampled class distribution:")
            print(pd.Series(y_train_resampled).value_counts(normalize=True))

            # Combine features and target for Bayesian Network
            train_data_resampled = pd.concat([pd.DataFrame(X_train_resampled, columns=X.columns), 
                                              pd.Series(y_train_resampled, name='Admit')], axis=1)

            # Create fully connected model
            nodes = list(train_data_resampled.columns)
            edges = create_fully_connected_model(nodes)
            model = BayesianNetwork(edges)

            # Train the model
            model.fit(train_data_resampled, estimator=BayesianEstimator, prior_type="BDeu")

            # Perform inference
            inference = VariableElimination(model)

            # Evaluate the model
            y_true = y_test
            y_pred = []

            for _, row in X_test.iterrows():
                evidence = {col: int(row[col]) for col in X_test.columns}
                try:
                    pred = inference.map_query(['Admit'], evidence=evidence)
                    y_pred.append(pred['Admit'])
                except IndexError:
                    # If an error occurs, predict the most common class
                    y_pred.append(y_true.mode()[0])

            accuracy = accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

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
                'Accuracy': accuracy,
                'Balanced Accuracy': balanced_accuracy,
                'F1 Score': f1,
                'Model': 'Fully Connected BN with SMOTE',
                'Model Structure': model_structure,
                'Marginal Probabilities': json.dumps(marginal_probs),
                'Classification Report': json.dumps(classification_report(y_true, y_pred, output_dict=True))
            })

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Classification Report:")
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
print(results_df[['GRE bins', 'TOEFL bins', 'GPA bins', 'Accuracy', 'Balanced Accuracy', 'F1 Score']])

# Find best configuration
best_result = results_df.loc[results_df['F1 Score'].idxmax()]
print("\nBest Configuration:")
print(best_result[['GRE bins', 'TOEFL bins', 'GPA bins', 'Accuracy', 'Balanced Accuracy', 'F1 Score']])

# Save results to CSV
results_df.to_csv('fully_connected_bn_model_results_with_smote.csv', index=False)
print("\nResults saved to 'fully_connected_bn_model_results_with_smote.csv'")
