import logging
import os
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)

async def run_agent(agent, num_steps, run_log_dir, send_game_updates, claude_logger):
    try:
        logger.info(f"Starting agent for {num_steps} steps")
        steps_completed = 0
        
        while steps_completed < num_steps:
            # Check if we should pause
            while getattr(agent.app.state, 'is_paused', False):
                await asyncio.sleep(0.1)
                continue
            
            # Get the current frame
            frame = agent.get_frame()
            
            # Save frame as PNG
            frame_count = steps_completed + 1
            frame_path = os.path.join(run_log_dir, "frames", f"frame_{frame_count:05d}.png")
            with open(frame_path, "wb") as f:
                f.write(frame)
            
            # Get Claude's message
            message = agent.get_last_message()
            
            # Log Claude's message
            if message:
                claude_logger.info(message)
            
            # Send updates to web clients
            await send_game_updates(frame, message)
            
            # Run one step
            agent.step()
            steps_completed += 1
            
            # Add a small delay to control the update rate
            await asyncio.sleep(0.1)
            
        logger.info(f"Agent completed {steps_completed} steps")
    except asyncio.CancelledError:
        logger.info("Agent task was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise 