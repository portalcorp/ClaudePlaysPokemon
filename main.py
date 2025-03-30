import argparse
import logging
import os
import asyncio
import uvicorn
from web.app import app
from agent.simple_agent import SimpleAgent
from contextlib import asynccontextmanager
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create a unique log directory for this run
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_log_dir = os.path.join(logs_dir, f"run_{current_time}")
os.makedirs(run_log_dir, exist_ok=True)
os.makedirs(os.path.join(run_log_dir, "frames"), exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(run_log_dir, "game.log"))
    ],
)

logger = logging.getLogger(__name__)

# Create a separate logger for Claude's messages
claude_logger = logging.getLogger("claude")
claude_logger.setLevel(logging.INFO)
claude_handler = logging.FileHandler(os.path.join(run_log_dir, "claude_messages.log"))
claude_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
claude_logger.addHandler(claude_handler)

@asynccontextmanager
async def lifespan(app):
    # Store log directory and claude logger in app state
    app.state.run_log_dir = run_log_dir
    app.state.claude_logger = claude_logger
    app.state.is_paused = False  # Initialize pause state
    
    # Startup: create agent but don't start it yet
    args = app.state.args
    agent = SimpleAgent(
        rom_path=args.rom_path,
        headless=True,
        sound=False,
        max_history=args.max_history,
        app=app  # Pass the app instance
    )
    app.state.agent = agent
    app.state.agent_task = None

    # Load save state if provided
    if args.save_state_path:
        try:
            agent.emulator.load_state(args.save_state_path)
            logger.info(f"Loaded save state from {args.save_state_path}")
        except Exception as e:
            logger.error(f"Failed to load save state: {e}")
            return
    
    yield
    # Shutdown: cleanup
    if hasattr(app.state, 'agent_task') and app.state.agent_task:
        app.state.agent_task.cancel()
        try:
            await app.state.agent_task
        except asyncio.CancelledError:
            pass
    if hasattr(app.state, 'agent'):
        app.state.agent.stop()

app.router.lifespan_context = lifespan

def main():
    parser = argparse.ArgumentParser(description="Claude Plays Pokemon - Web Version")
    parser.add_argument(
        "--rom", 
        type=str, 
        default="pokemon.gb",
        help="Path to the Pokemon ROM file"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1000, 
        help="Number of agent steps to run"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the web server on"
    )
    parser.add_argument(
        "--max-history", 
        type=int, 
        default=30, 
        help="Maximum number of messages in history before summarization"
    )
    parser.add_argument(
        "--save-state",
        type=str,
        help="Path to a save state file to load"
    )
    
    args = parser.parse_args()
    
    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom)
    else:
        rom_path = args.rom
    
    # Get absolute path to save state if provided
    save_state_path = None
    if args.save_state:
        if not os.path.isabs(args.save_state):
            save_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_state)
        else:
            save_state_path = args.save_state
        
        # Check if save state exists
        if not os.path.exists(save_state_path):
            logger.error(f"Save state file not found: {save_state_path}")
            return
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return
    
    # Store ROM path and other args in app state
    args.rom_path = rom_path
    args.save_state_path = save_state_path
    app.state.args = args
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()