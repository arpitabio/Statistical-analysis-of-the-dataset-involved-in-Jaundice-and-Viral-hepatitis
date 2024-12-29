# In[8]:


get_ipython().system('pip install statsmodels')


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\45 statistics.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df)

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Step 4: Histograms
plt.figure(figsize=(16, 10))

# Histogram for Age
plt.subplot(4, 4, 1)
sns.histplot(df['Age'], bins=10, kde=True, color='blue', edgecolor='black')
plt.title('Age Distribution')

# Histograms for Enzyme Parameters
for i in range(15):
    plt.subplot(4, 4, i + 2)
    sns.histplot(df.iloc[:, i + 1], bins=10, kde=True, color='green', edgecolor='black')
    plt.title(f'Biochemical Marker {i + 1}')

plt.tight_layout()
plt.show()

# Step 5: Correlation matrix
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[3]:


#Box-Cox plot, Q-Q plot
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\45 statistics.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df.head())

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Print the column names to check for discrepancies
print("\nColumn Names:")
print(df.columns)

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Columns to transform (excluding the 'disease' columns)
columns_to_transform = [
    'Bilirubin Direct', 'Bilirubin Indirect' 'Bilirubin Total', 'SGPT/ALT', 'SGOT/AST', 'ALP',
    'Total Protein', 'Cholesterol Total', 'LDL Cholesterol', 'HDL Cholesterol', 'Albumin', 'Globulin', 'A G ratio'
    'Triglycerides', 'VLDL Cholesterol'
]

# Function to perform Box-Cox transformation and Q-Q plot
def boxcox_qqplot(column):
    if df[column].min() <= 0:
        # Add a constant to make the data positive
        df[column] = df[column] - df[column].min() + 1
    transformed_data, lambda_value = stats.boxcox(df[column])

    print(f"\nLambda value for Box-Cox transformation of {column}: {lambda_value}")

    # Plotting the original and transformed data
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, color='blue', edgecolor='black')
    plt.title(f'Original {column} Distribution')

    plt.subplot(1, 2, 2)
    sns.histplot(transformed_data, kde=True, color='green', edgecolor='black')
    plt.title(f'Transformed {column} Distribution (Box-Cox)')

    plt.tight_layout()
    plt.show()

    

# Iterate over each column and apply the Box-Cox transformation and Q-Q plots
for column in columns_to_transform:
    if column in df.columns:
        print(f"Processing {column}...")
        boxcox_qqplot(column)
    else:
        print(f"Column {column} not found in the DataFrame.")


# In[2]:


#Box-Cox plot, Q-Q plot extended
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\45 statistics.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df.head())

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Print the column names to check for discrepancies
print("\nColumn Names:")
print(df.columns)

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Columns to transform (excluding the 'disease' columns)
columns_to_transform = [
    'Bilirubin Indirect','Bilirubin Total','Triglycerides','A G ratio'
]

# Function to perform Box-Cox transformation and Q-Q plot
def boxcox_qqplot(column):
    if df[column].min() <= 0:
        # Add a constant to make the data positive
        df[column] = df[column] - df[column].min() + 1
    transformed_data, lambda_value = stats.boxcox(df[column])

    print(f"\nLambda value for Box-Cox transformation of {column}: {lambda_value}")

    # Plotting the original and transformed data
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, color='blue', edgecolor='black')
    plt.title(f'Original {column} Distribution')

    plt.subplot(1, 2, 2)
    sns.histplot(transformed_data, kde=True, color='green', edgecolor='black')
    plt.title(f'Transformed {column} Distribution (Box-Cox)')

    plt.tight_layout()
    plt.show()

    

# Iterate over each column and apply the Box-Cox transformation and Q-Q plots
for column in columns_to_transform:
    if column in df.columns:
        print(f"Processing {column}...")
        boxcox_qqplot(column)
    else:
        print(f"Column {column} not found in the DataFrame.")


# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance, to_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import os
import graphviz  # Ensure graphviz is installed and accessible

# Load the data from the Excel file
file_path = r'C:\Users\gangu\OneDrive\Documents\Copy of 45 stat new.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

# Attempt to read the Excel file
try:
    data = pd.read_excel(file_path)
except PermissionError:
    raise PermissionError(f"Permission denied for file at path {file_path}. Please close the file if it's open in another application.")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# If there are missing values, fill them with the mean of the column
data = data.fillna(data.mean())

# Separate features (X) and target (y)
# Assuming the rightmost 4 columns are the targets (diseases)
X = data.iloc[:, :]


# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Box Plot for each feature separately with adjusted scales
features = X.columns
for feature in features:
    plt.figure(figsize=(10, 6))
    
    # Use log scale if the range of the feature is too large
    if data[feature].max() > 1000:
        sns.boxplot(data=np.log1p(data[feature]))
        plt.title(f'Box Plot for {feature} (Log Scale)')
        plt.yscale('log')
    else:
        sns.boxplot(data=data[feature])
        plt.title(f'Box Plot for {feature}')
    
    plt.show()


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\45 statistics.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df.head())

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Print the column names to check for discrepancies
print("\nColumn Names:")
print(df.columns)

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Columns to transform (excluding the 'disease' columns)
columns_to_transform = [
    'Bilirubin Indirect','Bilirubin Total','Triglycerides','A G ratio'
]

# Function to perform Box-Cox transformation and Q-Q plot
def boxcox_qqplot(column):
    if df[column].min() <= 0:
        # Add a constant to make the data positive
        df[column] = df[column] - df[column].min() + 1
    transformed_data, lambda_value = stats.boxcox(df[column])

    print(f"\nLambda value for Box-Cox transformation of {column}: {lambda_value}")

    return df[column], transformed_data

# Combine plots in a grid layout
def combined_plots(columns, df):
    num_cols = 2  # Number of columns in the plot grid
    num_rows = int(np.ceil(len(columns) * 2 / num_cols))  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 3))

    for i, column in enumerate(columns):
        if column in df.columns:
            original_data, transformed_data = boxcox_qqplot(column)
            print(f"Processing {column}...")

            # Original distribution plot
            sns.histplot(original_data, kde=True, color='blue', edgecolor='black', ax=axes[i*2 // num_cols, (i*2) % num_cols])
            axes[i*2 // num_cols, (i*2) % num_cols].set_title(f'Original {column} Distribution')

            # Transformed distribution plot
            sns.histplot(transformed_data, kde=True, color='green', edgecolor='black', ax=axes[i*2 // num_cols, (i*2 + 1) % num_cols])
            axes[i*2 // num_cols, (i*2 + 1) % num_cols].set_title(f'Transformed {column} Distribution (Box-Cox)')
        else:
            print(f"Column {column} not found in the DataFrame.")

    plt.tight_layout()
    plt.show()

combined_plots(columns_to_transform, df)


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\45 statistics.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df.head())

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Print the column names to check for discrepancies
print("\nColumn Names:")
print(df.columns)

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Columns to transform (excluding the 'disease' columns)
columns_to_transform = [
    'Bilirubin Direct', 'Bilirubin Indirect', 'Bilirubin Total', 'SGPT/ALT', 'SGOT/AST', 'ALP',
    'Total Protein'
]

# Function to perform Box-Cox transformation and return the transformed data
def boxcox_transform(column):
    if df[column].min() <= 0:
        # Add a constant to make the data positive
        df[column] = df[column] - df[column].min() + 1
    transformed_data, lambda_value = stats.boxcox(df[column])
    print(f"\nLambda value for Box-Cox transformation of {column}: {lambda_value}")
    return df[column], transformed_data

# Determine the grid size for subplots
num_plots = len(columns_to_transform) * 2
num_cols = 4
num_rows = int(np.ceil(num_plots / num_cols))

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
axes = axes.flatten()

# Plot each column's original and transformed data
plot_index = 0
for column in columns_to_transform:
    if column in df.columns:
        original_data, transformed_data = boxcox_transform(column)
        print(f"Processing {column}...")

        # Plot original data
        sns.histplot(original_data, kde=True, color='blue', edgecolor='black', ax=axes[plot_index])
        axes[plot_index].set_title(f'Original {column} Distribution')
        plot_index += 1

        # Plot transformed data
        sns.histplot(transformed_data, kde=True, color='green', edgecolor='black', ax=axes[plot_index])
        axes[plot_index].set_title(f'Transformed {column} Distribution (Box-Cox)')
        plot_index += 1
    else:
        print(f"Column {column} not found in the DataFrame.")

# Remove any unused subplots
for i in range(plot_index, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()
plt.show()

# Save the plot as an image file
fig.savefig('combined_boxcox_plots.png')


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\45 statistics.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df.head())

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Print the column names to check for discrepancies
print("\nColumn Names:")
print(df.columns)

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Columns to transform (excluding the 'disease' columns)
columns_to_transform = [
     'Cholesterol Total', 'LDL Cholesterol', 'HDL Cholesterol', 'Albumin', 'Globulin', 
    'A G ratio', 'Triglycerides', 'VLDL Cholesterol'
]

# Function to perform Box-Cox transformation and return the transformed data
def boxcox_transform(column):
    if df[column].min() <= 0:
        # Add a constant to make the data positive
        df[column] = df[column] - df[column].min() + 1
    transformed_data, lambda_value = stats.boxcox(df[column])
    print(f"\nLambda value for Box-Cox transformation of {column}: {lambda_value}")
    return df[column], transformed_data

# Determine the grid size for subplots
num_plots = len(columns_to_transform) * 2
num_cols = 4
num_rows = int(np.ceil(num_plots / num_cols))

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
axes = axes.flatten()

# Plot each column's original and transformed data
plot_index = 0
for column in columns_to_transform:
    if column in df.columns:
        original_data, transformed_data = boxcox_transform(column)
        print(f"Processing {column}...")

        # Plot original data
        sns.histplot(original_data, kde=True, color='blue', edgecolor='black', ax=axes[plot_index])
        axes[plot_index].set_title(f'Original {column} Distribution')
        plot_index += 1

        # Plot transformed data
        sns.histplot(transformed_data, kde=True, color='green', edgecolor='black', ax=axes[plot_index])
        axes[plot_index].set_title(f'Transformed {column} Distribution (Box-Cox)')
        plot_index += 1
    else:
        print(f"Column {column} not found in the DataFrame.")

# Remove any unused subplots
for i in range(plot_index, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()
plt.show()

# Save the plot as an image file
fig.savefig('combined_boxcox_plots.png')


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
file_path_1 = r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx'
file_path_2 = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'

# Read the datasets
data1 = pd.read_excel(file_path_1)
data2 = pd.read_excel(file_path_2)

# Merge the datasets
data = pd.concat([data1, data2], axis=0, ignore_index=True)

# Check for missing values
data = data.fillna(data.mean())

# Define the diseases
diseases = ['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis', 'Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice']

# Ensure the target columns are binary
for disease in diseases:
    data[disease] = data[disease].apply(lambda x: 1 if x > 0 else 0)

# Define the algorithms
algorithms = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Gaussian Naive Bayes': GaussianNB()
}

# Dictionary to store the evaluation metrics
metrics = {alg: {disease: {} for disease in diseases} for alg in algorithms}

# Function to evaluate models
def evaluate_model(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_true, y_prob[:, 1])
    }

# Separate features and target for each disease and evaluate each algorithm
for disease in diseases:
    print(f"Processing {disease}...")
    X = data.iloc[:, :-len(diseases)]
    y = data[disease]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for alg_name, alg in algorithms.items():
        print(f"Evaluating {alg_name}...")
        alg.fit(X_train, y_train)
        y_pred = alg.predict(X_test)
        y_prob = alg.predict_proba(X_test)
        metrics[alg_name][disease] = evaluate_model(y_test, y_pred, y_prob)

# Convert the metrics dictionary to a DataFrame
metrics_df = pd.DataFrame.from_dict({(i, j): metrics[i][j] 
                           for i in metrics.keys() 
                           for j in metrics[i].keys()},
                       orient='index').reset_index()
metrics_df.columns = ['Algorithm', 'Disease', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Plotting the evaluation metrics
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
for metric in metrics_to_plot:
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Disease', y=metric, hue='Algorithm', data=metrics_df)
    plt.title(f'Comparison of {metric.capitalize()} across Different Algorithms and Diseases')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# In[ ]:




