## Machine Learning and More
This repository is for me to work through different problems faced in statistics, machine learning, and more. For example, I have the following directories:
- [Bayesian Data Analysis](./bayesian_data_analysis/) where I work through the textbook of the same name, authored by Gelman.
- [Bayesian statistics tutorials](./bayesian_stats_tutorials/) where I work through some Bayesian statistics challenges that I was unclear on, and wanted to understand more of the theory, computation, and how to explain the results to a stakeholder.
<!-- - [Reinforcement Learning](./reinforcement_learning/) where I work on problems like how to set up a Markov reward/decision process, set up RL algorithms from scratch, model different problems in life (like Snakes and Ladders, the Frog Puzzle, studying the inventory dynamics of a small bicycle store, etc.). I wanted to strengthen my knowledge of numerical methods, quantitative modeling, and RL. -->


## ⚙️ Setting Up Your Environment

To ensure a consistent and isolated environment for running the code in this repository, it's highly recommended to use a **virtual environment**. This practice helps avoid conflicts with system-wide Python packages.

---

### 1. Create and Activate the Virtual Environment

We will use the built-in `venv` module to create the environment and name it `.venv`.

| Step | Command (Linux/macOS) | Command (Windows PowerShell) |
| :--- | :--- | :--- |
| **Create** | `python3 -m venv .venv` | `python -m venv .venv` |
| **Activate** | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |

You'll know the environment is active when you see `(.venv)` prepended to your command line prompt.

---

### 2. Install Dependencies with `uv`

Once your virtual environment is active, use **`uv`** (a fast Python package installer and resolver) to install the project dependencies.

#### A. Install Core Dependencies

These are the libraries listed under `dependencies` and `project.optional-dependencies` (e.g., `numpy`, `scipy`, `loguru`).

```bash
uv sync --all-extras
```

### 💻 Submitting Code (Pull Requests)

We use a **two-branch strategy** (`dev` and `main`). All contributions must first be merged into the **`dev`** branch for testing and review before being promoted to `main`.

#### 1. Fork, Clone, and Prepare

1.  **Fork** the repository on GitHub by clicking the "Fork" button.
2.  **Clone** your fork locally:
    ```bash
    git clone [https://github.com/YourUsername/machine-learning-and-more.git](https://github.com/YourUsername/machine-learning-and-more.git)
    ```
3.  **Navigate** into the new directory and ensure you are working from the **`dev`** branch:
    ```bash
    cd machine-learning-and-more
    git checkout dev
    ```

#### 2. Create a Feature Branch

Always create a new, descriptive branch off of **`dev`** for your contribution.

```bash
# Example: Creating a new feature branch
git checkout -b feature/your-awesome-feature dev