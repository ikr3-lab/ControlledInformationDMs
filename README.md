# DM Retrieval

DM Retrieval is a Python project for dealing with controlled information release and retrieval in Data Marketplaces. 

This is the repository linked to the article: Cassani, L., Livraga, G., Viviani, M. (2024). Assessing Document Sanitization for Controlled Information Release and Retrieval in Data Marketplaces. In: Goeuriot, L., et al. Experimental IR Meets Multilinguality, Multimodality, and Interaction. CLEF 2024. Lecture Notes in Computer Science, vol 14958. Springer, Cham. https://doi.org/10.1007/978-3-031-71736-9_4

## Installation

Use the package manager [poetry](https://pypi.org/project/poetry/) to install DM Retrieval dependencies.

```bash
poetry install
```

## Dataset
URL:
https://trec.nist.gov/data/core2018.html


## Project Structure

```bash
├── analysis/
├── processing/
└── summarization/
└── utils/
```

### Analysis 
The analysis directory contains scripts used for data exploration and preliminary investigation.

### Processing
The processing directory handles all data cleaning, conversion, and sanitization. It contains scripts used for Named Entity Recognition sanitization and unmasking phase.

### Summarization
The summarization directory contains summarization algorithms.

### Utils
The utils directory contains utils script. 
- doc.py : Doc class useful for modeling each document.
- experiments.py : Scripts used to perform the experiments.
- trec_parser.py : Scripts used to parse TREC dataset.
