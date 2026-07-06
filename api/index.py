import os
import sys

# Add root directory to sys.path so modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

# Export app for Vercel Serverless handler
handler = app
