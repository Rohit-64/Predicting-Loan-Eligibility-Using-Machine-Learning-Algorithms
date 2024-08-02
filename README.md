<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Prediction Using Machine Learning</title>
</head>
<body>
    <h1>Loan Eligibility Prediction Using Machine Learning</h1>

  <p>This project aims to predict loan eligibility using machine learning algorithms. It involves data preprocessing, exploratory data analysis (EDA), and building various machine learning models to achieve accurate predictions.</p>

  <h2>Table of Contents</h2>
    <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#exploratory-data-analysis">Exploratory Data Analysis</a></li>
        <li><a href="#data-preprocessing">Data Preprocessing</a></li>
        <li><a href="#model-building">Model Building</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#conclusion">Conclusion</a></li>
    </ul>

   <h2 id="introduction">Introduction</h2>
    <p>Loan eligibility prediction is crucial for financial institutions to assess the risk associated with lending. This project utilizes various machine learning techniques to predict whether an applicant is eligible for a loan based on historical data.</p>

   <h2 id="dataset">Dataset</h2>
    <p>
        <ul>
            <li><code>loan-train.csv</code>: Training dataset containing loan applicant information and their eligibility status.</li>
            <li><code>loan-test.csv</code>: Test dataset used to evaluate the model's performance.</li>
        </ul>
    </p>
    <p>The datasets include features such as:</p>
    <ul>
        <li>Loan Id</li>
      <li>Gender</li>
      <li>Marital Status</li>
      <li>Self Employed</li>
      <li>Dependents</li>
  <li>Applicant Income</li>
        <li>Coapplicant Income</li>
        <li>Loan Amount</li>
        <li>Loan Amount Term</li>
        <li>Credit History</li>
        

<li>Education</li>
        
<li>Property Area</li>
<li>Loan status</li>
  </ul>

   <h2 id="requirements">Requirements</h2>
    <p>To run this project, you need the following libraries:</p>
    <ul>
        <li>pandas</li>
        <li>numpy</li>
        <li>matplotlib</li>
        <li>seaborn</li>
        <li>scikit-learn</li>
    </ul>
    <p>You can install these libraries using pip:</p>
    <pre><code>pip install pandas numpy matplotlib seaborn scikit-learn</code></pre>

   <h2 id="exploratory-data-analysis">Exploratory Data Analysis</h2>
    <p>EDA is performed to understand the data distribution, identify patterns, and detect anomalies. Key steps include:</p>
    <ul>
        <li>Visualizing the distribution of features</li>
        <li>Analyzing correlations between features</li>
        <li>Handling missing values</li>
    </ul>

   <h2 id="data-preprocessing">Data Preprocessing</h2>
    <p>Data preprocessing steps include:</p>
    <ul>
        <li>Handling missing values</li>
        <li>Encoding categorical variables using Label Encoding</li>
        <li>Feature scaling</li>
    </ul>

  <h2 id="model-building">Model Building</h2>
    <p>We build and evaluate several machine learning models, including:</p>
    <ul>
        <li>Decision Tree Classifier</li>
        <li>Logistic Regression</li>
    </ul>
    <p>Model selection and hyperparameter tuning are done using cross-validation techniques.</p>

  <h2 id="evaluation">Evaluation</h2>
    <p>Model performance is evaluated using metrics such as:</p>
    <ul>
        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall</li>
        <li>F1 Score</li>
        <li>Confusion Matrix</li>
    </ul>
    <p>The model achieves an accuracy of 90%.</p>

  <h2 id="conclusion">Conclusion</h2>
    <p>The project successfully predicts loan eligibility using machine learning models. The best-performing model is selected based on evaluation metrics and is further optimized for better accuracy.</p>
</body>
</html>
