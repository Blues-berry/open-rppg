# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'open-rppg'
copyright = '2026, Kegang Wang'
author = 'Kegang Wang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# docs/source/conf.py

import os
import sys
# 将项目根目录加入路径，以便 Sphinx 能找到你的代码并提取 docstrings
sys.path.insert(0, os.path.abspath('../../')) 

project = 'Open-rppg'
release = '0.1.0'

# --- 核心扩展配置 ---
extensions = [
    'sphinx.ext.autodoc',       # 自动从代码提取文档
    'sphinx.ext.napoleon',      # 支持 Google/NumPy 风格的 docstrings
    'sphinx.ext.viewcode',      # 链接到源代码
    'myst_parser',              # 支持 Markdown
    'sphinx.ext.intersphinx',
]

# --- 国际化 (i18n) 配置 ---
# 告诉 Sphinx 翻译文件在哪里
locale_dirs = ['locale/']   
gettext_compact = False     

# --- 主题配置 ---
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# 避免 mock 导入错误（如果在构建环境中没装 jax/keras 可以取消注释）
autodoc_mock_imports = ["jax", "keras", "numpy", "cv2", "onnxruntime", "scipy", "av", "heartpy"]