#!/usr/bin/env python
"""Entry point for the llms.txt content analysis pipeline.

    uv run analyze.py --help

"""
import sys

from llmstxt_analysis.cli import main

if __name__ == "__main__":
    sys.exit(main())
