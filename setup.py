import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name = "src",
    version = "0.0.1",
    author = "parul2903",
    author_email = "parul3kin@gmail.com",
    description = "A small package for ANN implementation",
    long_description = long_description, # long description shows readme.md files 
    long_description_content_type = "text/markdown",
    url = "https://github.com/parul2903/ANN-implementation",
    packages = ["src"],
    python_requires = ">=3.8",
    install_requires = [
        "tensorflow",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "PyYAML"
    ]
)