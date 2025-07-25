

# Enhancing Graph Database Interaction Through Generative AI-Driven Natural Language Interface for Financial-Fraud Detection

This repository contains the implementation of a financial fraud detection system integrating **Graph Databases (Neo4j)** with **Generative AI models** such as **LLaMA-2**, **T5**, **BERT**, and **GPT-4**, enabling a **Natural Language Interface** for interacting with graph-based transaction data.

---

## 🧠 Project Overview

Traditional financial fraud detection systems struggle with complex, relational data and user interaction. This project addresses these challenges by:

- Using **Neo4j Graph Database** to represent relationships between accounts and transactions.
- Enabling **Natural Language Querying** powered by generative AI models (LLaMA-2, T5, GPT-4).
- Combining **rule-based filtering** and **AI-based classification** for fraud detection.
- Visualizing graph results interactively using **Pyvis**.

---

pip install -r requirements.txt

streamlit run app.py


 Paper Publication
This project is based on our accepted research paper:

"Enhancing Graph Database Interaction Through Generative AI-Driven Natural Language Interface for Financial-Fraud Detection"

https://ieeexplore.ieee.org/document/10725408


🛠️ Technologies Used
Technology	
Python -	Core development language,
Streamlit -	UI framework for interactive web apps,
Neo4j	- Graph database for transaction storage,
Pyvis -	Network visualization library,
Transformers -	HuggingFace library for LLMs (LLaMA2, T5),
Scikit-learn -	ML models and preprocessing,
Pandas	- Data manipulation



✅ Project Outcomes
Improved fraud detection using both AI and rule-based logic.

Natural language-based interaction made querying intuitive.

Effective graph-based visualization helped trace fraud patterns.

Higher accuracy achieved with LLaMA-2 for financial transaction data.
