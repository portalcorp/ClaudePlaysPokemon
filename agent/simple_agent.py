import base64
import copy
import io
import logging
import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE
from agent.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT
from agent.tools import AVAILABLE_TOOLS

from agent.emulator import Emulator
from anthropic import Anthropic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()



class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, app=None, use_overlay=False):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            app: FastAPI app instance for state management
            use_overlay: Whether to show tile overlay visualization
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        self.anthropic_client = Anthropic()
        self.google_client = genai.Client()
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.last_message = "Game starting..."  # Initialize last message
        self.app = app  # Store reference to FastAPI app
        self.use_overlay = use_overlay  # Store overlay preference
        
        # Modify system prompt if overlay is enabled
        if use_overlay:
            self.system_prompt = SYSTEM_PROMPT + '''
            There is a color overlay on the tiles that shows the following:

            🟥 Red tiles for walls/obstacles
            🟩 Green tiles for walkable paths
            🟦 Blue tiles for NPCs/sprites
            🟨 Yellow tile for the player with directional arrows (↑↓←→)
            '''
        else:
            self.system_prompt = SYSTEM_PROMPT

    def get_frame(self) -> bytes:
        """Get the current game frame as PNG bytes.
        
        Returns:
            bytes: PNG-encoded screenshot of the current frame with optional tile overlay
        """
        screenshot = self.emulator.get_screenshot_with_overlay() if self.use_overlay else self.emulator.get_screenshot()
        # Convert PIL image to PNG bytes
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return buffered.getvalue()

    def get_last_message(self) -> str:
        """Get Claude's most recent message.
        
        Returns:
            str: The last message from Claude, or a default message if none exists
        """
        return self.last_message

    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")
            
            result = self.emulator.press_buttons(buttons, wait)
            
            # Get a fresh screenshot after executing the buttons with tile overlay
            screenshot = self.emulator.get_screenshot_with_overlay()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")
            
            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get a fresh screenshot after executing the navigation with tile overlay
            screenshot = self.emulator.get_screenshot_with_overlay()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Navigation result: {result}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def step(self):
        """Execute a single step of the agent's decision-making process."""
        try:
            messages = copy.deepcopy(self.message_history)

            if len(messages) >= 3:
                if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                
                if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                    messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

            # Get model response
            response = self.anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=messages,
                tools=AVAILABLE_TOOLS,
                temperature=TEMPERATURE,
            )

            # Update last message with Claude's response text
            self.last_message = next((block.text for block in response.content if block.type == "text"), self.last_message)

            logger.info(f"Response usage: {response.usage}")

            # Extract tool calls
            tool_calls = [
                block for block in response.content if block.type == "tool_use"
            ]

            # Display the model's reasoning
            for block in response.content:
                if block.type == "text":
                    logger.info(f"[Text] {block.text}")
                elif block.type == "tool_use":
                    logger.info(f"[Tool] Using tool: {block.name}")

            # Process tool calls
            if tool_calls:
                # Add assistant message to history
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({"type": "tool_use", **dict(block)})
                
                self.message_history.append(
                    {"role": "assistant", "content": assistant_content}
                )
                
                # Process tool calls and create tool results
                tool_results = []
                for tool_call in tool_calls:
                    tool_result = self.process_tool_call(tool_call)
                    tool_results.append(tool_result)
                
                # Add tool results to message history
                self.message_history.append(
                    {"role": "user", "content": tool_results}
                )

                # Check if we need to summarize the history
                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

        except Exception as e:
            logger.error(f"Error in agent step: {e}")
            raise

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                self.step()
                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                raise e

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info(f"[Agent] Generating conversation summary...")
        
        # Get a new screenshot for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Create messages for the summarization request - pass the entire conversation history
        messages = copy.deepcopy(self.message_history) 


        if len(messages) >= 3:
            if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            
            if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SUMMARY_PROMPT,
                    }
                ],
            }
        ]
        
        # Get summary from Claude
        response = self.anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=self.system_prompt,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        # Extract the summary text
        summary_text = " ".join([block.text for block in response.content if block.type == "text"])
        
        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        
        logger.info(f"[Agent] Message history condensed into summary.")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()