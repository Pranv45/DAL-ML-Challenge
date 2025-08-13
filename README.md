# Medical Chart Embedding Multi-label Classification

This project aims to build a multi-label classification model to predict ICD10 codes for medical chart embeddings. The model is optimized for the average micro-F2 score.

## Project Structure

The notebook contains the following sections:

1.  **Data Loading and Preprocessing**: Loading the embedding and label files, concatenating them, and converting the text labels into a multi-hot encoded vector representation.
2.  **Exploratory Data Analysis (EDA) and Visualization**: Analyzing label frequencies, distribution of labels per chart, and visualizing dimensionality-reduced embeddings.
3.  **Data Splitting**: Splitting the data into training and validation sets.
4.  **Model Selection**: Discussion on potential model architectures and the chosen approach (Neural Network).
5.  **Model Training**: Training the selected neural network model on the training data.
6.  **Model Evaluation**: Evaluating the trained model on the validation set using the micro-F2 score.
7.  **Prediction on Test Data**: Loading the test data and generating predictions.
8.  **Generate Submission File**: Formatting predictions into the required submission CSV format.
9.  **Report**: A detailed report summarizing the entire process, including data engineering, EDA, model details, evaluation results, and any hacks or workarounds.

## Data

The data is provided in the following files:

*   `embedings_1.npy`: First chunk of medical chart embeddings.
*   `icd_codes_1.txt`: Corresponding ICD10 labels for `embedings_1.npy`.
*   `embedings_2.npy`: Second chunk of medical chart embeddings.
*   `icd_codes_2.txt`: Corresponding ICD10 labels for `embedings_2.npy`.
*   `test_data.npy`: Medical chart embeddings for which predictions need to be generated.
*   `sample_solution.csv`: A sample file showing the required submission format.

**Note**: Ensure these data files are in the same directory as the notebook or provide the correct file paths in the code.


### Handling Large Data Files with Git LFS

The `.npy` and `.txt` files in this project can be large, making them unsuitable for standard Git due to storage and performance limits.
**Git Large File Storage (LFS)** handles this by replacing large files with lightweight pointers in your Git history, while storing the actual file content separately on a remote server.

#### Setup and Usage

```bash
# Install Git LFS (only once per system)
git lfs install

# Navigate to your repository root (if not already there)
cd /path/to/your/repo

# Track large file types
git lfs track "*.npy"
git lfs track "*.txt"

# Add and commit changes
git add .
git commit -m "Add data files with Git LFS"

# Push to remote repository
git push origin main   # Replace 'main' with your branch name if different

'''

## Setup and Running the Code

### Prerequisites

*   Python 3.6+
*   Jupyter Notebook or Google Colab
*   Required Python libraries:
    *   `numpy`
    *   `pandas`
    *   `sklearn`
    *   `tensorflow`
    *   `matplotlib`
    *   `seaborn`

You can install the required libraries using pip: