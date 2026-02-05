# 📦 REQUIREMENTS.TXT INSTALLATION GUIDE

---

## 🎯 WHAT IS REQUIREMENTS.TXT?

A `requirements.txt` file is a standard way to list all Python packages needed for a project. It allows you to:

✅ **Install all packages with one command**  
✅ **Share exact dependencies with team members**  
✅ **Reproduce the environment on different machines**  
✅ **Version control your dependencies**  
✅ **Deploy to production with confidence**

---

## 📋 AVAILABLE REQUIREMENTS FILES

I've created **4 different versions** for different use cases:

### 1️⃣ **requirements.txt** (STANDARD - RECOMMENDED)
- Complete package list with minimum versions
- Best for: Local development, general use
- Size: ~3-5 GB
- Time: 15-25 minutes

### 2️⃣ **requirements_minimal.txt** (QUICK START)
- Only essential packages
- Best for: Quick testing, learning
- Size: ~1-2 GB
- Time: 5-10 minutes

### 3️⃣ **requirements_colab.txt** (GOOGLE COLAB)
- Optimized for Google Colab environment
- Skips pre-installed packages
- Best for: Running in Colab
- Time: 10-15 minutes

### 4️⃣ **requirements_production.txt** (PRODUCTION)
- Exact pinned versions
- Best for: Production deployment, reproducibility
- Guarantees same versions everywhere
- Time: 15-25 minutes

---

## 🚀 HOW TO USE REQUIREMENTS.TXT

### **Method 1: Standard Installation** (RECOMMENDED)

```bash
# Step 1: Create virtual environment (HIGHLY RECOMMENDED)
python -m venv rag_env

# Step 2: Activate virtual environment
# On Linux/Mac:
source rag_env/bin/activate

# On Windows:
rag_env\Scripts\activate

# Step 3: Upgrade pip
pip install --upgrade pip

# Step 4: Install from requirements.txt
pip install -r requirements.txt

# Step 5: Verify installation
pip list
```

---

### **Method 2: Google Colab Installation**

```python
# In a Colab notebook cell:
!pip install -r requirements_colab.txt

# Fix SQLite issue for ChromaDB
import sys
!pip install pysqlite3-binary
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
```

---

### **Method 3: Without Virtual Environment** (NOT RECOMMENDED)

```bash
# Direct installation (may conflict with system packages)
pip install -r requirements.txt
```

---

### **Method 4: Quick Start (Minimal)**

```bash
# For quick testing
pip install -r requirements_minimal.txt
```

---

## ⚙️ INSTALLATION OPTIONS

### **Option A: Install Everything (Standard)**
```bash
pip install -r requirements.txt
```
✅ Complete setup  
❌ Takes 15-25 minutes  
❌ Downloads 3-5 GB

---

### **Option B: Install with Upgrade**
```bash
# Upgrade existing packages to meet requirements
pip install -r requirements.txt --upgrade
```
✅ Updates old packages  
⚠️ May break existing projects

---

### **Option C: Install Without Dependencies**
```bash
# Install only listed packages, skip their dependencies
pip install -r requirements.txt --no-deps
```
⚠️ Advanced use only  
❌ May cause import errors

---

### **Option D: Install with Progress Bar**
```bash
pip install -r requirements.txt --verbose
```
✅ See detailed installation progress

---

## 🔍 VERIFY INSTALLATION

### **Method 1: Check Installed Packages**
```bash
pip list
```

### **Method 2: Verify Specific Package**
```bash
pip show langchain
```

### **Method 3: Run Verification Script**
```python
import sys

packages_to_check = [
    'numpy',
    'pandas',
    'langchain',
    'chromadb',
    'sentence_transformers',
    'openai',
]

print("🔍 Verifying Packages...\n")
missing = []

for pkg in packages_to_check:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - MISSING")
        missing.append(pkg)

if not missing:
    print("\n🎉 All packages installed successfully!")
else:
    print(f"\n⚠️ Missing packages: {', '.join(missing)}")
    print("Run: pip install -r requirements.txt")
```

---

## 🛠️ CREATING YOUR OWN REQUIREMENTS.TXT

### **Method 1: Export Current Environment**
```bash
# Export all installed packages
pip freeze > requirements.txt
```

### **Method 2: Export Only Project Packages** (RECOMMENDED)
```bash
# Use pipreqs to scan your code
pip install pipreqs
pipreqs /path/to/your/project
```

### **Method 3: Manual Creation**
```bash
# Create file manually
nano requirements.txt

# Add packages line by line:
langchain==0.1.9
chromadb>=0.4.22
sentence-transformers
```

---

## 🔧 VERSION PINNING STRATEGIES

### **Strategy 1: Exact Version (Production)**
```txt
langchain==0.1.9
```
✅ Reproducible  
❌ No security updates

### **Strategy 2: Minimum Version (Development)**
```txt
langchain>=0.1.9
```
✅ Get updates  
⚠️ May break compatibility

### **Strategy 3: Compatible Version (Balanced)**
```txt
langchain~=0.1.9
```
✅ Get patch updates (0.1.10, 0.1.11)  
❌ No minor updates (0.2.0)

### **Strategy 4: No Version (Flexible)**
```txt
langchain
```
✅ Always latest  
❌ May break anytime

---

## 🐛 TROUBLESHOOTING

### **Error 1: "Could not find a version that satisfies the requirement"**
```bash
# Solution 1: Update pip
pip install --upgrade pip

# Solution 2: Try older version
pip install langchain==0.1.0

# Solution 3: Remove version constraint
# Edit requirements.txt: langchain>=0.1.0 → langchain
```

---

### **Error 2: "No matching distribution found"**
```bash
# Solution: Check package name spelling
# Wrong: langchain_openai
# Correct: langchain-openai
```

---

### **Error 3: "Permission denied"**
```bash
# Solution: Use --user flag
pip install -r requirements.txt --user

# Or use sudo (Linux/Mac)
sudo pip install -r requirements.txt
```

---

### **Error 4: "Conflicting dependencies"**
```bash
# Solution 1: Use --force-reinstall
pip install -r requirements.txt --force-reinstall

# Solution 2: Create fresh virtual environment
rm -rf rag_env
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
```

---

### **Error 5: ChromaDB SQLite Error (Colab)**
```python
# Add this BEFORE importing chromadb
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
```

---

## 📊 INSTALLATION SIZE BREAKDOWN

| Component | Size | Time |
|-----------|------|------|
| Core (numpy, pandas) | ~500 MB | 2-3 min |
| LangChain packages | ~200 MB | 2-3 min |
| Transformers | ~1-2 GB | 5-8 min |
| Sentence-transformers | ~500 MB | 3-5 min |
| ChromaDB | ~100 MB | 1-2 min |
| PyTorch (if not installed) | ~2 GB | 5-10 min |
| **TOTAL** | **~3-5 GB** | **15-25 min** |

---

## 🎯 RECOMMENDED WORKFLOW

### **For New Project:**
```bash
# 1. Create project directory
mkdir my_rag_project
cd my_rag_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Copy requirements.txt to project
cp /path/to/requirements.txt .

# 4. Install packages
pip install -r requirements.txt

# 5. Create .gitignore
echo "venv/" > .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore

# 6. Initialize git
git init
git add requirements.txt .gitignore
git commit -m "Initial commit with requirements"
```

---

### **For Existing Project:**
```bash
# 1. Navigate to project
cd my_existing_project

# 2. Export current requirements
pip freeze > requirements_old.txt

# 3. Copy new requirements
cp /path/to/requirements.txt .

# 4. Install/upgrade
pip install -r requirements.txt --upgrade

# 5. Test your code
python test_imports.py

# 6. If everything works, commit
git add requirements.txt
git commit -m "Update requirements"
```

---

## 🌐 ENVIRONMENT-SPECIFIC TIPS

### **Local Development (Windows/Mac/Linux)**
```bash
# Use virtual environment ALWAYS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Google Colab**
```python
# In notebook cell
!pip install -r requirements_colab.txt

# Add SQLite fix
import sys
!pip install pysqlite3-binary
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
```

### **Docker Container**
```dockerfile
# In Dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

### **Production Server**
```bash
# Use pinned versions
pip install -r requirements_production.txt

# For security updates
pip list --outdated
```

---

## 📝 BEST PRACTICES

### ✅ DO:
- Use virtual environments
- Pin versions in production
- Keep requirements.txt in git
- Document special installation steps
- Test after installing requirements
- Update regularly for security

### ❌ DON'T:
- Install globally (use venv)
- Mix pip and conda
- Include system packages
- Ignore version conflicts
- Commit virtual environment folder
- Use `pip freeze` blindly (includes all deps)

---

## 🎓 ADVANCED TIPS

### **Tip 1: Separate Dev and Prod Requirements**
```txt
# requirements_base.txt (common)
langchain>=0.1.0
chromadb>=0.4.22

# requirements_dev.txt (development only)
-r requirements_base.txt
pytest
black
flake8

# requirements_prod.txt (production)
-r requirements_base.txt
gunicorn
```

### **Tip 2: Use pip-compile (pip-tools)**
```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in (high-level)
echo "langchain" > requirements.in
echo "chromadb" >> requirements.in

# Generate requirements.txt (with all deps)
pip-compile requirements.in

# Install
pip-sync requirements.txt
```

### **Tip 3: Speed Up Installation**
```bash
# Use parallel downloads
pip install -r requirements.txt --use-feature=fast-deps

# Use cache
pip install -r requirements.txt --cache-dir=/tmp/pip-cache
```

---

## 🚀 QUICK REFERENCE

### **One-Line Installation**
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### **One-Line Colab Installation**
```python
!pip install -r requirements_colab.txt && pip install pysqlite3-binary
```

### **One-Line Verification**
```bash
python -c "import langchain, chromadb, sentence_transformers; print('✅ All OK')"
```

---

## 📞 WHEN TO USE WHICH FILE

| Scenario | File to Use |
|----------|-------------|
| First time setup | `requirements.txt` |
| Quick testing | `requirements_minimal.txt` |
| Google Colab | `requirements_colab.txt` |
| Production deployment | `requirements_production.txt` |
| Sharing with team | `requirements.txt` |
| CI/CD pipeline | `requirements_production.txt` |
| Learning/Tutorial | `requirements_minimal.txt` |

---

## 🎉 SUMMARY

**YES, you can save packages in requirements.txt!**

✅ **Advantages:**
- One command installation
- Version control
- Reproducible environments
- Easy sharing
- Professional standard

✅ **Recommendation:**
- Use `requirements.txt` for standard setup
- Use `requirements_minimal.txt` for quick start
- Use `requirements_production.txt` for deployment
- ALWAYS use virtual environments

✅ **Installation:**
```bash
pip install -r requirements.txt
```

---

**END OF INSTALLATION GUIDE**
