import streamlit as st
from st_paywall import add_auth
import requests
import pytz
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from collections import Counter
from serpapi import GoogleSearch
import pytrends
from googleapiclient.discovery import build
from openai import OpenAI
from streamlit_js import st_js, st_js_blocking
from google_auth_oauthlib.flow import Flow
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
import plotly.express as px
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from datetime import datetime
from dateutil import parser
import pandas as pd
import numpy as np
import json
import re

st.set_page_config(layout="wide")
st.title("Youtube Viral Chatbot 🚀")

add_auth(required=True)

# ONLY AFTER THE AUTHENTICATION + SUBSCRIPTION, THE USER WILL SEE THIS ⤵
# The email and subscription status is stored in session state.
st.write(f"Subscription Status: {st.session_state.user_subscribed}")
st.write("🎉 Yay! You're all set and subscribed! 🎉")
st.write(f'By the way, your email is: {st.session_state.email}')
