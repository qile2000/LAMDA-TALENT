from setuptools import setup, find_packages

setup(
    name='TALENT',
    version='0.0.2',
    description='TALENT: A Tabular Analytics and Learning Toolbox',
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author='TALENT Contributors from LAMDA-Tabular Group',
    author_email='lsy00043@gmail.com',
    packages=find_packages(),
    keywords='pytorch tabular deep learning machine learning',
    url='https://github.com/qile2000/LAMDA-TALENT/tree/talent-pip',
    install_requires=[
        # "annoy==1.17.3",
        "tqdm==4.66.4",
        "category_encoders==2.6.3",
        "delu==0.0.23",
        "einops==0.8.0",
        "numpy==1.26.4",
        "optuna==3.6.1",
        "pandas==2.2.2",
        "qhoptim==1.1.0",
        "Requests==2.31.0",
        "scikit_learn==1.4.2",
        "scipy==1.13.0",
        "torch==2.0.1",
        # xgboost==2.0.3
        # lightgbm==4.3.0
        # catboost==1.2.3
    ],
    include_package_data=True,
    # python_requires='==3.10',
)