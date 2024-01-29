import streamlit as st
import collections
import functools
import inspect
import textwrap
import numpy as np


import pandas as pd


df = pd.read_csv('ICD9_descriptions', sep="\t", header=None)

print(df)