# Automated-Machine-Learning-App
The Automated ML WebApp project is a tool developed using Python, Pandas Profiling, PyCaret, and Streamlit frameworks. It aims to simplify and streamline the process of building machine learning models by automating various tasks involved in data preprocessing, feature engineering, model selection, and downloading the trained models.

# Website
![](https://github.com/praj2408/Automated-Machine-Learning-App/blob/main/docs/bandicam%202023-06-23%2016-47-03-898%20(1).gif)

## Frameworks Used:
### [Python](https://www.python.org/)
The core of the web application is built using Python, a versatile programming language widely used in the field of data science and machine learning. Python provides a rich ecosystem of libraries and frameworks that make it an ideal choice for building such applications.

### [Ydata-Profiling (Pandas-Profiling)](https://pypi.org/project/pandas-profiling/)
Pandas Profiling is an essential component of this project, as it enables automatic exploratory data analysis. With Pandas Profiling, users can quickly gain insights into their datasets, including data types, missing values, statistical summaries, correlations, and more. This helps users understand the characteristics and quality of their data, allowing them to make informed decisions during the model-building process.

### [Pycaret](https://pycaret.org/)
PyCaret, another crucial component, provides an automated machine-learning environment. It offers a high-level API that simplifies complex machine-learning tasks such as feature selection, hyperparameter tuning, model training, and evaluation. PyCaret supports a wide range of machine learning algorithms and automatically selects the best models based on user-defined evaluation metrics.

### [Streamlit](https://streamlit.io/)
The web application leverages the Streamlit framework to create an interactive and user-friendly interface. Streamlit allows developers to design custom web applications with minimal effort. It enables users to upload their datasets, visualize data profiles generated by Pandas Profiling, select target variables and features, choose machine learning algorithms, and configure hyperparameters for model training.

The Automated ML WebApp project combines the power of these tools to provide users with an intuitive and efficient solution for automating the end-to-end process of machine learning. It eliminates the need for manual coding and extensive domain expertise, making it accessible to both novice and experienced data scientists. With this web application, users can save time and effort in building, evaluating, and deploying machine learning models, empowering them to focus more on the interpretation and application of the results.

## How to Run locally:
1. clone the repository:
```bash
git clone https://github.com/praj2408/Automated-Machine-Learning-App
```
2. Create a new virtual environment:
```bash
conda create -n automl python==3.9 -y
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Run the project:
```bash
streamlit run app.py
```

## Contributions
Contributions to this project are welcome! To contribute, please follow the standard GitHub workflow for pull requests.

## Contact information
If you have any questions or comments about this project, feel free to contact the project maintainer at [gmail](prajwalgbdr03@gmail.com.)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
