# Parkinson's Disease Detection Using Voice Analysis
## A Machine Learning Approach to Early Diagnosis

## Abstract

This project explores the feasibility of using machine learning algorithms to detect Parkinson's disease through voice analysis. I implemented three different classification models (Logistic Regression, SVM, and Random Forest) on the UCI Parkinson's dataset to compare their performance. The results show that voice-based biomarkers can achieve high accuracy in distinguishing between healthy individuals and Parkinson's patients, with Random Forest performing best at ~95% accuracy.

## Introduction

Parkinson's disease affects millions worldwide, and early detection is crucial for effective treatment. Traditional diagnosis relies on clinical examination, which can be subjective and time-consuming. Recent research suggests that voice changes occur early in Parkinson's disease progression, making voice analysis a promising non-invasive screening tool.

I was inspired by papers showing that Parkinson's patients exhibit measurable changes in voice characteristics like jitter, shimmer, and fundamental frequency variations. This project aims to build a machine learning system that can automatically detect these changes.

## Dataset

I used the Parkinson's Disease Classification dataset from UCI Machine Learning Repository (Little et al., 2007). The dataset contains voice recordings from 31 people, with 23 having Parkinson's disease and 8 being healthy controls. Each recording was analyzed to extract 22 voice features.

**Dataset Details:**
- Total samples: 195 (multiple recordings per person)
- Features: 22 voice measurements
- Classes: Healthy (0) vs Parkinson's (1)
- Class distribution: ~75% Parkinson's, ~25% Healthy

The features include:
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Jitter(%) - Measures of frequency variation
- MDVP:Shimmer - Measures of amplitude variation
- HNR - Harmonics-to-noise ratio
- RPDE - Recurrence period density entropy
- DFA - Detrended fluctuation analysis
- And 16 other voice characteristics

## Methodology

### Data Preprocessing
1. Loaded the dataset and checked for missing values (none found)
2. Separated features from target variable
3. Split data into 80% training and 20% testing sets
4. Applied StandardScaler for SVM and Logistic Regression

### Model Selection
I chose three different algorithms to compare:

1. **Logistic Regression**: Simple linear classifier, good baseline
2. **Support Vector Machine (SVM)**: Effective for high-dimensional data
3. **Random Forest**: Ensemble method, handles non-linear relationships well

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.2% | 88.1% | 97.4% | 92.5% | 0.89 |
| SVM | 89.7% | 90.2% | 97.4% | 93.7% | 0.91 |
| **Random Forest** | **94.9%** | **95.1%** | **97.4%** | **96.2%** | **0.95** |

Random Forest achieved the best performance across all metrics. The most important features were:
1. MDVP:Jitter(%) - 0.15
2. MDVP:Shimmer - 0.12
3. HNR - 0.11
4. RPDE - 0.10
5. DFA - 0.09

## Discussion

The results are promising! Random Forest's high accuracy (94.9%) suggests that voice analysis could be a viable screening tool for Parkinson's disease. The most discriminative features align with clinical knowledge - jitter and shimmer are known indicators of voice disorders.

However, there are limitations:
- Small dataset size (195 samples)
- Imbalanced classes (more Parkinson's patients than healthy)
- No demographic information (age, gender, disease severity)
- Single recording per person (could benefit from multiple samples)

## Future Work

- Collect larger, more diverse dataset
- Include demographic features
- Try deep learning approaches
- Implement real-time voice analysis
- Validate on different populations
- Add audio file upload functionality

## Technical Implementation

### Files Structure
```
├── parkinsons_analysis.py    # Main analysis script
├── app.py                   # Streamlit web demo
├── parkinsons.data          # UCI dataset
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Dependencies
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit
- joblib

### How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run analysis: `python parkinsons_analysis.py`
3. Launch web app: `streamlit run app.py`

## AI Tools Used

I used several AI tools to help with this project:
- **ChatGPT**: Helped with code debugging and explaining ML concepts
- **Grok**: Assisted with data analysis and visualization ideas
- **GitHub Copilot**: Code completion and suggestions
- **Claude**: Helped structure the research documentation

These tools were invaluable for learning and implementing the project, but all analysis and conclusions are my own.

## References

1. Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A. E., & Moroz, I. M. (2007). Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection. *Biomedical Engineering Online*, 6(1), 23.

2. Tsanas, A., Little, M. A., McSharry, P. E., Spielman, J., & Ramig, L. O. (2012). Novel speech signal processing algorithms for high-accuracy classification of Parkinson's disease. *IEEE Transactions on Biomedical Engineering*, 59(5), 1264-1271.

3. Sakar, B. E., Isenkul, M. E., Sakar, C. O., Sertbas, A., Gurgen, F., Delil, S., ... & Kursun, O. (2013). Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings. *IEEE Journal of Biomedical and Health Informatics*, 17(4), 828-834.

4. Rusz, J., Hlavnička, J., Tykalová, T., Bušková, J., Klempíř, J., Sonka, K., ... & Růžička, E. (2016). Smartphone allows capture of speech abnormalities associated with early risk of Parkinson's disease. *IEEE Transactions on Biomedical Engineering*, 65(5), 1025-1032.

## Fun Disclaimer 🚨

**WARNING: This is a student project for learning purposes only!**

- Do NOT use this for actual medical diagnosis
- Do NOT replace doctor visits with this tool
- Do NOT trust AI predictions with your health
- This is just a fun ML experiment, not a medical device
- If you think you have Parkinson's, go see a real doctor!
- I'm just a student, not a medical professional
- This project is probably full of bugs anyway 😅

**Seriously, don't use this for real medical decisions. I'm not responsible if you do!**

---

## Acknowledgments

Thanks to:
- UCI ML Repository for the dataset
- My Professor(Chatgpt, grok, claude) for guidance
- ChatGPT, Grok, and Copilot for being helpful coding buddies
- The open-source community for amazing tools
- Anyone who reads this and doesn't judge my code too harshly

---

*This project was created as part of my machine learning coursework. It's not perfect, but I learned a lot building it!*
