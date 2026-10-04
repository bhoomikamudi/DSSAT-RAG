"""Local development launcher.

On Windows, uvicorn's default ProactorEventLoop is incompatible with async
psycopg, so every database query fails. The loop is created before the app is
imported, so the selector loop must be chosen here, before uvicorn starts.

Usage (from backend/):  python run_dev.py [--port 8005] [--reload] [--llm-model NAME]

--llm-model overrides OPENAI_MODEL for this process only (.env is not
changed). On the UF gateway the OpenAI-routed models (gpt-4o, gpt-4.1,
gpt-5) currently have their connections reset, while the Claude models
respond, so development verification uses e.g. --llm-model claude-4.6-sonnet.
"""
import argparse
import asyncio
import os
import sys
import warnings

import uvicorn

if sys.platform == "win32":
    with warnings.catch_warnings():
        # Event loop policies are deprecated in Python 3.14 but still honored.
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--llm-model", help="override OPENAI_MODEL for this process")
    args = parser.parse_args()

    if args.llm_model:
        # Environment variables take precedence over .env in the settings,
        # and the reload worker inherits them.
        os.environ["OPENAI_MODEL"] = args.llm_model
        print(f"LLM model override for this process: {args.llm_model}")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
