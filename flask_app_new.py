#!/usr/bin/env python3
"""
LoRA Craft Flask Server - Entry Point

A web-based interface for GRPO (Group Relative Policy Optimization) fine-tuning.
This is the main entry point that uses the application factory pattern.
"""

import os
from app_factory import create_app

if __name__ == '__main__':
    # Create the Flask app and SocketIO instance
    app, socketio = create_app()

    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))

    print(f"=" * 80)
    print(f"LoRA Craft Server")
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"=" * 80)

    # Run the application with SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to True for development
        allow_unsafe_werkzeug=True  # Required for SocketIO in development
    )
