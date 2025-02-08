# 🚀 Contributing to Verify Me

Thank you for your interest in contributing to this project! We welcome contributions from everyone, whether it's fixing bugs, improving documentation, or adding new features. This guide will help you get started.

---
## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](https://github.com/chutrunganh/Biometric_IT4432E/blob/master/.github/CODE_OF_CONDUCT.md). Please read it before making any contributions. 🙌

## 🤝 How to Contribute

🐛 **Reporting Bugs**

- **Check Existing Issues**: Before reporting a bug, please check the  to ensure it hasn’t already been reported.

- **Create a New Issue**: If the bug hasn’t been reported, open a new issue and provide the following details:

  - A clear and descriptive title.
  
  - Steps to reproduce the issue.
  
  - Expected vs. actual behavior.
  
  - Screenshots, logs, or error messages (if applicable).
  
  - Your environment (e.g., OS, browser, version).

💡 **Suggesting Enhancements**

- **Check Existing Discussions**: Look through the [Issues page](https://github.com/chutrunganh/Biometric_IT4432E/issues) to see if your enhancement has already been suggested.

- **Open a New Issue**: If not, create a new issue and include:

  - A clear and descriptive title.

  - A detailed explanation of the enhancement.

  - Why this enhancement would be useful.

  - Examples or references (if applicable).

📤 **Submitting Pull Requests**

1. **Fork the Repository**
   
    Start by forking the repository to your GitHub account.

2. **Clone the Repository**

   Clone the forked repository to your local machine:

    ```bash
    git clone https://github.com/chutrunganh/Biometric_IT4432E.git
    ```

3. **Create a Branch**

    Create a new branch for your feature or bug fix:

    ```bash
    git checkout -b feature/your-feature-name
    ```

4. **Make Changes**

    Make your changes to the codebase. Ensure your code follows the project's coding standards and conventions.

5. **Test Your Changes**

    Ensure your changes work as expected and do not introduce new issues.

6. **Commit Changes**

     Commit your changes with a clear and concise commit message:

    ```bash
    git add .
    git commit -m "Add feature: your feature name"
    ```

7. **Push Changes**

     Push your changes to your forked repository:

    ```bash
    git push origin feature/your-feature-name
    ```

8. **Create a Pull Request**

    Go to the original repository on GitHub and create a pull request from your forked repository. 

    Provide a clear title and description for your PR, including:

    - The purpose of the changes.
    
    - Any related issues (e.g., "Fixes #123").
    
    - Screenshots or test results (if applicable)

---
## ⚒️ Development Setup

1. **Clone the repo**
```bash
git clone https://github.com/chutrunganh/Biometric_IT4432E.git
```

2. **Install dependencies**

Navigate to the project folder:

```bash
cd REPLACE_WITH_YOUR_PATH/Biometric_IT4432E
```

- With Linux

```bash
# Activate python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install pip tool if you haven't already
sudo pacman -Syu base-devel python-pip # With Arch-based
# sudo apt update && sudo apt upgrade -y && sudo apt install build-essential python3-pip  # With Debian-based, use this command instead
pip install --upgrade pip setuptools wheel

# Install all required dependencies 
pip install -r requirements_for_Linux.txt
```

- With Windows

```bash
python -m venv venv
.\venv\Scripts\activate.bat # If execute in CMD
# .\venv\Scripts\activate.ps1 # If execute in PowerShell

# Install pip tool if you haven't already
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

# Install all required dependencies 
pip install -r requirements_for_Windosn.txt

# Install ipykernel in your virtual environment
pip install ipykernel 
python -m ipykernel install --user --name=venv --display-name "Python (venv)" # Create a new kernel for Jupyter

```

Choose the kernel named `venv` when running Jupyter Notebook.

It may take about 15-30 minutes to download all dependencies, depending on your internet speed.


> [!IMPORTANT]  
> This project requires **Python 3.12.x**. Some other versions, such as 3.10.x, have been reported to have compatibility issues with dependencies.

3. **Follow the code files**

Follow the code files from 1 to 4 (you can choose to just follow Pipeline1 or Pipeline2) and read the instructions, run the code inside these files to generate and process data. Note that this is a pipeline, so do not skip any files; otherwise, errors will occur due to missing files.

---

## 🎨 Style Guidelines

- **Code Formatting**: Follow the existing code style (e.g., indentation, naming conventions).

- **Documentation**: Update documentation (e.g., README, comments) to reflect your changes.

- **Testing**: Write unit tests for new features or bug fixes.
---
## ❓ Questions or Need Help?

If you have any questions or need assistance, feel free to:

- Open an [Issue](https://github.com/chutrunganh/Biometric_IT4432E/issues/new?template=Blank+issue).

- Reach out to us via this [Email](mailto:chutrunganh04@gmail.com).

---

We appreciate your contributions and look forward to collaborating with you!  🎉 🎉 🎉
