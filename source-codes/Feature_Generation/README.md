
# Feature Extraction with PaDEL-Descriptor and RDKit

This repository contains two Jupyter notebooks for feature extraction from molecular structures using PaDEL-Descriptor and RDKit. These tools are commonly used in cheminformatics for generating molecular descriptors and fingerprints, which are essential for various machine learning applications in drug discovery and toxicity prediction.

## Files

1. **Padel_Feature.ipynb**
2. **RDKIT_feature.ipynb**

## Requirements

Before running the notebooks, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- PaDEL-Descriptor (requires Java)
- RDKit
- Pandas
- NumPy
- Matplotlib

You can install the necessary Python packages using pip:

```bash
pip install pandas numpy matplotlib
```

For RDKit, you can follow the installation instructions [here](https://www.rdkit.org/docs/Install.html).

## Notebooks Overview

### 1. Padel_Feature.ipynb

This notebook demonstrates how to use the PaDEL-Descriptor tool to generate molecular descriptors and fingerprints from SMILES strings.

#### Steps:
1. **Loading SMILES Data**: Reads SMILES strings from a CSV file.
2. **Setting Up PaDEL-Descriptor**: Configures the PaDEL-Descriptor Java tool.
3. **Generating Descriptors**: Executes PaDEL-Descriptor to generate 2D and 3D molecular descriptors.
4. **Handling Timeout**: Implements a timeout mechanism to handle molecules that take too long to process.
5. **Saving Features**: Saves the generated features to a CSV file.

### 2. RDKIT_feature.ipynb

This notebook demonstrates how to use the RDKit library to generate molecular descriptors and fingerprints from SMILES strings.

#### Steps:
1. **Loading SMILES Data**: Reads SMILES strings from a CSV file.
2. **Generating Descriptors**: Uses RDKit to generate various molecular descriptors and fingerprints.
3. **Handling Errors**: Implements error handling for molecules that cannot be processed.
4. **Saving Features**: Saves the generated features to a CSV file.

## Output

The generated CSV files containing the molecular descriptors and fingerprints are stored in the same directory as the notebooks. Each notebook saves its output with a unique filename for easy identification.

- **Padel_Feature.ipynb**: The output file is saved as `padel_features.csv`.
- **RDKIT_feature.ipynb**: The output file is saved as `rdkit_features.csv`.


## creating mergered.csv
-Combining all these feature based on there Smiles string , we get our mererged.csv.This code is not been present here.
-On it contradicting lable(duplicate smiles string) have ben removed
