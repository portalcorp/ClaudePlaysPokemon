import base64
import copy
import io
import logging
import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, USE_NAVIGATOR

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


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Focus hard on the game screen and try to figure out the position of the tile you are trying to reach. Be very vigilant and try and spot important sprites.

There is a color overlay on the tiles that shows the following:

🟥 Red tiles for walls/obstacles
🟩 Green tiles for walkable paths
🟦 Blue tiles for NPCs/sprites
🟨 Yellow tile for the player with directional arrows (↑↓←→)

Try to enter Mt. Moon immediately, don't go into the Pokemon center. Exit the town immediately. Try to locate the cave in your very first screenshot.

You are starting at a pre-loaded save state. You are starting right outside of the Pewter City Pokemon center before you head to Mt. Moon. Your goal is to enter and then get out of Mt. Moon on Route 4.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands. Try to find the necessary ladders. The ladders sprites are visible faintly in the green blocks, they do not have their own distinct color.

Do not waste time picking up items or even talking to NPCs. You are trying to get out of Mt. Moon to Route 4 as quickly as possible.

You can also ask Google for information, and you will recieve a concise response that is based on search results.

Try your best not to backtrack your steps, but if you get stuck, you can ask Google for help. If you keep seeing the same screen or NPCs, you are probably stuck in a loop and should ask Google for help.

Don't waste time battling or trying to catch Pokemon. You are trying to get out of Mt. Moon to Route 4 as quickly as possible.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay.

Generally, you are pretty terrible at navigating through the game, so don't be afraid to try new things or think really hard about what you should do next."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Important decisions you've made
2. Current objectives or goals you're working toward
3. Any strategies or plans you've mentioned
4. How far away you are from your objective which is to get out of Mt. Moon to Route 4
5. Things you have already tried and what you have learned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


AVAILABLE_TOOLS = [
    {
        "name": "press_buttons",
        "description": "Press a sequence of buttons on the Game Boy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                    },
                    "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                }
            },
            "required": ["buttons"],
        },
    },
    {
        "name": "ask_google",
        "description": "Search Google using Gemini model and get a grounded response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to ask Google"
                }
            },
            "required": ["query"],
        },
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append({
        "name": "navigate_to",
        "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "The row coordinate to navigate to (0-8)."
                },
                "col": {
                    "type": "integer",
                    "description": "The column coordinate to navigate to (0-9)."
                }
            },
            "required": ["row", "col"],
        },
    })


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, app=None):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            app: FastAPI app instance for state management
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        self.client = Anthropic()
        self.google_client = genai.Client(api_key='AIzaSyBA6Ot3qdYfavwbQtB1CIY-MEz7myPSzGI')
        self.google_search_tool = Tool(google_search=GoogleSearch())
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.last_message = "Game starting..."  # Initialize last message
        self.app = app  # Store reference to FastAPI app

    def get_frame(self) -> bytes:
        """Get the current game frame as PNG bytes.
        
        Returns:
            bytes: PNG-encoded screenshot of the current frame with tile overlay
        """
        screenshot = self.emulator.get_screenshot_with_overlay()
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
        elif tool_name == "ask_google":
            query = tool_input["query"]
            logger.info(f"[Google Search] Querying: {query}")
            
            try:
                response = self.google_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=query,
                    config=GenerateContentConfig(
                        tools=[self.google_search_tool],
                        response_modalities=["TEXT"],
                    )
                )
                
                answer = response.candidates[0].content.parts[0].text
                grounding = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
                
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": [
                        {"type": "text", "text": f"Search query: {query}\n\nAnswer: {answer}\n\nGrounding sources:\n{grounding}"}
                    ],
                }
            except Exception as e:
                logger.error(f"Error in Google search: {e}")
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": [
                        {"type": "text", "text": f"Error performing Google search: {str(e)}"}
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
            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
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
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
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