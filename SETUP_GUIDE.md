# LMT Dimensionality Reduction Toolkit - Setup Guide

## 🚀 Quick Start

### Prerequisites Checklist
- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] Basic command line knowledge
- [ ] SQLite database from LMT experiments
- [ ] DB Browser for SQLite (optional but recommended)

## 📥 Installation Steps

### 1. Get the Code
```bash
# Clone the repository
git clone https://github.com/your-username/LMT-dim-reduction-toolkit
cd LMT-dim-reduction-toolkit
```

### 2. Set Up Python Environment
```bash
# Create a virtual environment (recommended)
python -m venv lmt_env

# Activate the environment
# On Windows:
.\lmt_env\Scripts\activate
# On Unix/MacOS:
source lmt_env/bin/activate
```

### 3. Install Dependencies
⚠️ **Important**: Follow this exact order
```bash
# 1. First, install numpy
pip install numpy==1.23.5

# 2. Then scipy
pip install scipy>=1.9.0

# 3. Finally, all other requirements
pip install -r requirements.txt
```

## 🔧 First-Time Setup

### Database Preparation
1. Create a dedicated folder for your databases
2. Copy your LMT SQLite databases to this folder
3. Make a backup of your original databases

### Verify Installation
```python
# Run the test script
python src/utils/test_setup.py
```

## 🏃 Running Your First Analysis

### 1. Event Filtering
```python
python src/preprocessing/event_filtered.py
```
You'll see a GUI where you need to:
- Select your database file
- Choose the experiment start time
- Wait for the filtering process to complete

### 2. Process Behaviors
```python
python src/behavior/behavior_processor.py
```
This creates the basic behavior statistics tables.

### 3. View Results
- Open DB Browser for SQLite
- Navigate to your database
- Check the newly created tables
- Look in data/behavior_stats_intervals_analyzed/ for visualizations

## 🔍 Troubleshooting

### Common Issues

#### 1. "Module not found" Errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

#### 2. Database Connection Issues
- Check file permissions
- Ensure path has no special characters
- Verify database isn't corrupted

#### 3. GUI Problems
- Windows: Reinstall Python with tkinter
- Linux: `sudo apt-get install python3-tk`
- Mac: Install Python through Homebrew

## 📊 Analysis Options

### Temporal Resolutions
1. **12-hour intervals** (Night Cycle)
   - Best for day/night comparisons
   - Default analysis mode

2. **4-hour chunks** (Circadian)
   - More detailed temporal patterns
   - Good for activity rhythm analysis

3. **Hourly resolution**
   - Finest temporal granularity
   - Detailed behavior tracking

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Development mode
export LMT_ENV=development
export LMT_DEBUG=true

# Production mode
export LMT_ENV=production
export LMT_DEBUG=false
```

### Custom Analysis Parameters
- Edit config files in `src/config/`
- Adjust thresholds in analysis scripts
- Modify visualization settings

## 📝 Best Practices

1. **Data Management**
   - Always backup databases
   - Use descriptive filenames
   - Keep original data separate

2. **Analysis Workflow**
   - Start with small datasets
   - Validate results frequently
   - Document parameter changes

3. **Resource Usage**
   - Close unused database connections
   - Clear large variables after use
   - Monitor memory usage

## 🆘 Getting Help

1. **Check Documentation**
   - Read the main README
   - Review this setup guide
   - Look at example scripts

2. **Common Solutions**
   - Restart Python environment
   - Clear cache and temporary files
   - Update all dependencies

3. **Contact Support**
   - Email: andreastivala.as@gmail.com
   - Include error messages
   - Describe your setup

## 🎯 Next Steps

After basic setup:
1. Try different temporal resolutions
2. Explore visualization options
3. Experiment with analysis parameters
4. Create custom analysis pipelines

Remember: Always validate results and maintain backups of your data! 