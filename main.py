import argparse
import hashlib
import json
import os
import random
import re
from typing import List, Dict, Optional, Tuple, Union

import litellm
import unicodedata
from litellm import completion
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
# Enable rich tracebacks for better error visualization
from rich.traceback import install

install()

# Constants and Configuration
MAX_TOKENS = 4096
PROMPT_DIRECTORIES = ["prompts"]

console = Console()

DEFAULT_PROMPT = "crispr"
DEFAULT_CONVERSATION_BASE = 'conversation'
DEFAULT_TAGS = {
    'n' : 10,
}

# short-hand abbreviations for the best models we want
MODEL_ABBREVIATIONS = {
    # 'l31-405b': 'fireworks_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
    'l31-405b': 'fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct',
    'l31-70b': 'fireworks_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'sonnet-35': 'claude-3-5-sonnet-20240620',
    'gpt-4om': 'gpt-4o-mini',
}

DEFAULT_MODEL = "claude-3-5-sonnet-20240620"

litellm.drop_params = True

# Read API keys from corresponding .env file in the working directory
with open('.env', 'r') as env_file:
    for line in env_file:
        key, value = line.strip().split('=')
        os.environ[key] = value.strip('"')

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

class Prompt:
    def __init__(self, content: Union[str, 'Prompt'], inputs: Dict[str, str] = None):
        self.inputs = inputs or {}
        self.capabilities = []

        self.content = content

        if isinstance(content, Prompt):
            self.content = content.content
            self.capabilities = content.capabilities
            self.inputs.update(content.inputs)
        else:
            try:
                self.content = self._load_file(content)
            except FileNotFoundError:
                self.content = self._process_content_static(content)

    def process_content_dynamic(self, conversation: 'Conversation') -> 'Prompt':
        def replace_dynamic(match):
            directive = match.group(1)
            parts = directive.split(':')
            func = parts[0]
            args = parts[1:]

            if func == 'func':
                return self._process_func(args, conversation)
            else:
                console.print(f"[bold red]Error:[/bold red] Unsupported dynamic directive: {{{{{directive}}}}}",
                              style="bold red")
                return ''

        new_content = re.sub(r'{{(func:[^{}]+)}}', replace_dynamic, self.content)
        p = Prompt(new_content, self.inputs)
        p.capabilities = list(self.capabilities)
        return p

    def _process_func(self, args: List[str], conversation: 'Conversation') -> str:
        func_name = args[0]
        func_args = args[1:]

        # Convert string arguments to their corresponding values
        processed_args = []
        for arg in func_args:
            if arg.isdigit():
                processed_args.append(int(arg))
            elif arg in self.inputs:
                processed_args.append(int(self.inputs[arg]))
            else:
                processed_args.append(arg)

        if func_name in ['random_message', 'message']:
            return self._func_message(conversation, *processed_args)
        elif func_name == 'messages':
            return self._func_messages(conversation, *processed_args)
        elif func_name == 'count':
            return self._func_count(conversation, *processed_args)
        elif func_name == 'last':
            return self._func_last(conversation, *processed_args)
        elif func_name == 'first':
            return self._func_first(conversation, *processed_args)
        elif func_name == 'date':
            return self._func_date(*processed_args)
        elif func_name == 'input':
            return self._func_input(*processed_args)
        elif func_name == 'choice':
            return self._func_choice(*processed_args)
        elif func_name == 'if':
            return self._func_if(*processed_args)
        else:
            console.print(f"[bold red]Error:[/bold red] Unsupported function: {func_name}", style="bold red")
            return ''

    def _func_message(self, conversation: 'Conversation', *args) -> str:
        if not conversation.messages:
            return ''
        if len(args) == 0:
            return random.choice(conversation.messages).content
        elif len(args) == 1:
            index = args[0] - 1  # Convert to 0-based index
            return conversation.messages[index].content if 0 <= index < len(conversation.messages) else ''
        elif len(args) == 2:
            start, end = args
            if start == 'min': start = 0
            if end == 'max': end = len(conversation.messages)
            messages = conversation.messages[start - 1:end]
            return random.choice(messages).content if messages else ''
        else:
            console.print("[bold red]Error:[/bold red] Invalid number of arguments for random_message/message function", style="bold red")
            return ''

    def _func_messages(self, conversation: 'Conversation', *args) -> str:
        if len(args) != 2:
            console.print("[bold red]Error:[/bold red] messages function requires 2 arguments", style="bold red")
            return ''
        start, end = args
        messages = conversation.messages[start - 1:end]
        return '\n'.join(msg.content for msg in messages)

    def _func_count(self, conversation: 'Conversation', *_) -> str:
        return str(len(conversation.messages))

    def _func_last(self, conversation: 'Conversation', *args) -> str:
        count = args[0] if args else 1
        return '\n'.join(msg.content for msg in conversation.messages[-count:])

    def _func_first(self, conversation: 'Conversation', *args) -> str:
        count = args[0] if args else 1
        return '\n'.join(msg.content for msg in conversation.messages[:count])

    def _func_date(self, *args) -> str:
        from datetime import datetime
        format_string = args[0] if args else "%Y-%m-%d %H:%M:%S"
        return datetime.now().strftime(format_string)

    def _func_input(self, *args) -> str:
        key = args[0] if args else None
        return self.inputs.get(key, '')

    def _func_choice(self, *args) -> str:
        return random.choice(args) if args else ''

    def _func_if(self, *args) -> str:
        if len(args) != 3:
            console.print("[bold red]Error:[/bold red] if function requires 3 arguments", style="bold red")
            return ''
        condition, true_value, false_value = args
        return true_value if condition.lower() in ('true', 'yes', '1') else false_value

    def _is_file(self, name: str) -> bool:
        # Check in prompt directories and current directory
        for directory in PROMPT_DIRECTORIES + [os.path.dirname(__file__)]:
            # Check for both name and name.txt
            if os.path.isfile(os.path.join(directory, name)) or os.path.isfile(os.path.join(directory, name + '.txt')):
                return True
        return False

    def _load_file(self, name: str) -> str:
        for directory in PROMPT_DIRECTORIES + [os.path.dirname(__file__)]:
            for file_name in [name, name + '.txt']:
                path = os.path.join(directory, file_name)
                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return self._process_content_static(f.read())

        raise FileNotFoundError(f"File not found: {name} or {name}.txt")

    def _process_content_static(self, content: str) -> str:
        # Replace inputs
        for k, v in self.inputs.items():
            content = content.replace(f"{{{{{k}}}}}", v)

        # Remove comments
        content = re.sub(r'{{#.*?}}', '', content)

        # Extract capabilities
        content = self._extract_capabilities(content)

        # Process {{...}} directives
        def process_match(match):
            directive = match.group(1)
            processed = self._process_directive(directive)
            # If the processed result is the same as the original directive, return it unchanged
            if processed == f"{{{{{directive}}}}}":
                return processed
            # Otherwise, return the processed result without the surrounding {{}}
            return processed

        # Use a while loop to ensure all nested directives are processed
        prev_content = None
        while prev_content != content:
            prev_content = content
            content = re.sub(r'{{(.*?)}}', process_match, content)

        return content.strip()

    def _process_directive(self, directive: str) -> str:
        if directive.startswith('func:'):
            console.print(f"[bold yellow]Ignoring directive:[/bold yellow] {{{{[italic]{directive}[/italic]}}}}",
                          style="bold yellow")
            return f'{{{{{directive}}}}}'
        elif directive in self.inputs:
            return self.inputs[directive]
        elif self._is_file(directive):
            return self._load_file(directive)
        else:
            console.print(f"[bold red]Error:[/bold red] Unsupported directive, file not found, and input not found: {{{{[italic]{directive}[/italic]}}}}",
                          style="bold red")
            raise ValueError(f"Unsupported directive or file not found: {{{{[italic]{directive}[/italic]}}}}")

    def _extract_capabilities(self, content: str) -> str:
        lines = content.split('\n')
        if lines:
            capabilities_pattern = r'^{{(.*?)}}'
            matches = re.findall(capabilities_pattern, lines[0])
            for match in matches:
                self.capabilities.extend([cap.strip() for cap in match.split(',')])
            lines[0] = re.sub(capabilities_pattern, '', lines[0])
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"Prompt({self.content[:50]}{'...' if len(self.content) > 50 else ''}, capabilities={self.capabilities})"

    def pretty_print(self):
        """
        Beautifully print the prompt content and capabilities using rich.
        """
        # Create a styled text for the content
        content_text = Text(self.content)
        content_text.stylize("cyan")

        # Create a panel for the content
        content_panel = Panel(
            content_text,
            title="[bold blue]Prompt Content[/bold blue]",
            border_style="blue",
            expand=False
        )

        # Create a styled text for the capabilities
        capabilities_text = Text("\n".join(f"• {cap}" for cap in self.capabilities))
        capabilities_text.stylize("green")

        # Create a panel for the capabilities
        capabilities_panel = Panel(
            capabilities_text,
            title="[bold green]Capabilities[/bold green]",
            border_style="green",
            expand=False
        )

        # Print the panels
        console.print(content_panel)
        console.print(capabilities_panel)

def fix_unicode(text):
    # Function to convert each match
    def convert(match):
        try:
            # Convert the hex value to an integer
            code_point = int(match.group(1), 16)
            # Convert the integer to a Unicode character
            return chr(code_point)
        except ValueError:
            # If conversion fails, return the original match
            return match.group(0)

    # Regex pattern to match Unicode escape sequences
    pattern = r'\\u([0-9a-fA-F]{4})'

    # Replace all occurrences of the pattern
    fixed_text = re.sub(pattern, convert, text)

    # Normalize the text to composed form (NFC)
    fixed_text = unicodedata.normalize('NFC', fixed_text)

    return fixed_text

class Conversation:
    def __init__(self, name, messages: List[Message]):
        self.name = name
        self.messages = messages

    @classmethod
    def from_file(cls, file_path: str) -> 'Conversation':
        """Load conversation from file, supporting JSON or Markdown formats."""
        with console.status("[bold green]Loading conversation...", spinner="dots"):
            if file_path.endswith('.json'):
                conversation = cls._load_from_json(file_path)
            elif file_path.endswith('.md'):
                conversation = cls._load_from_markdown(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Remove leading and trailing whitespace from all messages
            for message in conversation.messages:
                message.content = message.content.strip()
                message.content = fix_unicode(message.content)

            return conversation

    @classmethod
    def _load_from_json(cls, file_path: str) -> 'Conversation':
        name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in data):
                messages = [Message(msg["role"], msg["content"]) for msg in data]
                return cls(name, messages)
            elif isinstance(data, dict) and "chat_messages" in data:
                chat_messages = data.get("chat_messages", [])
                if all(isinstance(msg, dict) and "sender" in msg and "text" in msg for msg in chat_messages):
                    messages = [Message("user" if msg["sender"].lower() == "human" else "assistant", msg["text"]) for msg in chat_messages]
                    return cls(name, messages)
            raise ValueError("Unsupported JSON structure. Expected a list of messages with 'role' and 'content' keys, or the legacy format with 'chat_messages'.")

    @classmethod
    def _load_from_markdown(cls, file_path: str) -> 'Conversation':
        messages = []
        current_role = None
        current_content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Human:') or line.startswith('Assistant:'):
                    if current_role:
                        messages.append(Message(current_role, "".join(current_content).strip()))
                        current_content = []
                    current_role = 'user' if line.startswith('Human:') else 'assistant'
                else:
                    current_content.append(line)
        if current_role:
            messages.append(Message(current_role, "".join(current_content).strip()))
        return cls(os.path.basename(file_path), messages)

    def to_dict_list(self) -> List[Dict[str, str]]:
        return [msg.to_dict() for msg in self.messages]

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_dict_list(), f, indent=2)

    def print_conversation(self, max_message_length: Optional[int] = None):
        """Print compact, color-coded conversation in a box."""
        conversation_content = []
        for i,msg in enumerate(self.messages):
            role_color = "[cyan]" if msg.role == "assistant" else "[green]"
            role_display = "AI:" if msg.role == "assistant" else "Human:"
            content = msg.content
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length - 3] + "..."
            content = content.replace('\n', '\\n')  # Replace newlines with explicit \n characters
            conversation_content.append(f"{i}. {role_color}{role_display} {content}[/]")

        panel = Panel(
            "\n".join(conversation_content),
            title=f"{self.name}" if self.name else "Conversation",
            expand=False,
            border_style="bold",
            padding=(1, 1)
        )
        console.print(panel)

    def __str__(self):
        """Return a string representation of the conversation."""
        return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in self.messages])

    def __repr__(self):
        """Return a formal string representation of the conversation."""
        return f"Conversation(messages={self.messages})"

class ConversationRefactorer:
    """
    A module to refactor conversation.

    1. the execute method is called with a prompt and some k=v tags
    2. tags are plugged into the prompt where {{k}} becomes {{v}}
    3. prompt comments are set like this {{#this is a comment}} and are removed from the final inference messages
    4. capabilities are applied according to a comment at the start of the prompt (first line) which specifies it like so {{transform,crispr,...}}
    5. each capability runs sequentially over its own aggregated text
    6. updated ontology:
        - process -> capabilities
        - crispr -> (the output is initialized to the conversation history and is modified progressively according to targeted operations)
        - transform -> (the output of the refactor becomes the content inside the first detected ```code block``` in the aggregated result)
    """
    def __init__(self):
        self.capabilities = {
            'crispr': self._refactor_crispr,
            'rewrite': self._refactor_rewrite,
            'augment': self._refactor_augment
        }


    def __call__(self,
             prompt: str|Prompt,
             conversation: Conversation,
             max_aggregate_length: int = 999999999,
             max_tokens: int = 256,
             model=None,
             temperature: float = 0.7,
             top_k: int = None,
             top_p: float = 1.0,
             frequency_penalty: float = 0.0,
             presence_penalty: float = 0.0,
             dry: bool = False
             ) -> Conversation:

        prompt = Prompt(prompt)
        prompt = prompt.process_content_dynamic(conversation)
        prompt.pretty_print()

        if not prompt.capabilities:
            raise ValueError("No capabilities for the prompt.")

        messages = conversation.to_dict_list()
        messages.append({
            "role": "user",
            "content": prompt.content
        })


        if model and model in MODEL_ABBREVIATIONS:
            model = MODEL_ABBREVIATIONS[model]

        kwargs = {}
        if top_k: kwargs['top_k'] = top_k


        work_output = Conversation("output", conversation.messages[:-1])  # Initialize work output with original conversation, excluding the prompt
        local_output = Conversation("output_local", [])
        local_output.messages.append(Message("user", prompt.content))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        )
        task = progress.add_task(f"[cyan]{model}... ", total=None)

        console_output = []

        def get_renderable():
            return Group(
                Panel(process_aggregated_content, title="Aggregated Process Content", expand=False, border_style="bold"),
                progress,
                *console_output
            )
        
        if not dry:
            stream = completion(
                model=model or DEFAULT_MODEL,
                max_tokens=max_tokens,
                messages=messages,
                stream=True,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs,
            )

            process_aggregated_content = ""  # Store aggregated content for the entire process
            aggregated_content = {capability: "" for capability in self.capabilities}

            with Live(get_renderable(), console=console, refresh_per_second=4, auto_refresh=False) as live:
                for part in stream:
                    text = part.choices[0].delta.content
                    if text is None:
                        continue

                    process_aggregated_content += text or ""

                    for capability in prompt.capabilities:
                        if capability in self.capabilities:
                            aggregated_content[capability] += text
                    if any(len(content) > max_aggregate_length for content in aggregated_content.values()):
                        console_output.append(f"[yellow]Warning:[/yellow] Max aggregate length ({max_aggregate_length}) reached for at least one capability. Stopping stream.")
                        # Print the aggregated content that went over the limit
                        for cap, content in aggregated_content.items():
                            if len(content) > max_aggregate_length:
                                console_output.append(f"[bold]Aggregated content for {cap}:[/bold]")
                                console_output.append(content[:max_aggregate_length] + "...")
                        break

                    # Process capabilities sequentially
                    for capability in prompt.capabilities:
                        if capability in self.capabilities:
                            work_output, should_clear = self.capabilities[capability](work_output, aggregated_content[capability])
                            if should_clear:
                                aggregated_content[capability] = ""  # Clear aggregated content only if the capability indicates it should be cleared
                        else:
                            console_output.append(f"[yellow]Warning:[/yellow] Unknown capability '{capability}'")

                    live.update(get_renderable(), refresh=True)
    
            # Add the AI's reply (the generated content from this inference)
            local_output.messages.append(Message("assistant", process_aggregated_content))
        else:
            console.print("[yellow]Dry run enabled. Skipping inference and refactoring...[/yellow]")
        
        return work_output, local_output


    def _refactor_crispr(self, conversation: Conversation, content: str) -> Tuple[Conversation, bool]:
        lines = content.split('\n')
        changes_made = False

        # Combine lines that are part of the same entry
        combined_lines = []
        current_line = ""
        for line in lines:
            if re.match(r'^\d+\.', line):
                if current_line:
                    combined_lines.append(current_line)
                current_line = line
            else:
                current_line += " " + line.strip()
        if current_line:
            combined_lines.append(current_line)

        # print(content)

        for line in combined_lines:
            match = re.match(r'^(?:\d+\.)\s*"(.*?)"\s([→+-])\s"(.*?)"', line.strip(), re.MULTILINE)
            if match:
                # console.print(line)
                # console.print(match.groups())
                # console.print("")

                old, operation, new = match.groups()
                old = old.strip('"')
                new = new.strip('"')
                operation_successful = False

                if operation == '→':
                    changes_made = True
                    for msg in conversation.messages:
                        if re.search(re.escape(old), msg.content):
                            msg.content = re.sub(re.escape(old), new, msg.content)
                            operation_successful = True
                    if operation_successful:
                        console.print(f"[yellow]chg[/yellow]  {old}  →  [yellow]{new}[/yellow]")
                    else:
                        console.print(f"[red]Error:[/red] '{old}' not found for replacement.")
                elif operation == '+':
                    changes_made = True
                    for msg in conversation.messages:
                        if re.search(re.escape(old), msg.content):
                            msg.content = re.sub(re.escape(old), old + ' ' + new, msg.content)
                            operation_successful = True
                    if operation_successful:
                        console.print(f"[green]add[/green]  {old}  +  [green]{new}[/green]")
                    else:
                        console.print(f"[red]Error:[/red] '{old}' not found for addition.")


        return conversation, changes_made

    def _refactor_rewrite(self, conversation: Conversation, content: str) -> Tuple[Conversation, bool]:
        code_block_match = re.search(r'```(?:[\w]*\n)?(.*?)```', content, re.DOTALL)
        if code_block_match:
            transformed_content = code_block_match.group(1).strip()
            return Conversation([Message("assistant", transformed_content)]), True
        else:
            console.print("[yellow]Transform:[/yellow] No code block found in the response")
            return conversation, False

    def _refactor_augment(self, conversation: Conversation, content: str) -> Tuple[Conversation, bool]:
        # Implementation for augment refactoring
        # This is a placeholder and should be implemented based on the augment prompt
        console.print("[yellow]Augment:[/yellow] Placeholder implementation")
        return conversation, False
    

# ---------------------------------------------------------------------------------------------------------------


def get_latest_iteration(base_name):
    iteration = 1
    while os.path.exists(f"{base_name}.r{iteration:03d}.json"):
        iteration += 1
    return iteration - 1 if iteration > 0 else None

def get_conversation_file(base_name, iteration):
    return f"{base_name}.r{iteration:03d}.json" if iteration is not None else f"{base_name}.json"

def main():
    console.print(Panel.fit("[bold cyan]Conversation Refactoring Tool[/bold cyan]", border_style="bold"))

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Refactor a conversation using the Anthropic API.")
    parser.add_argument("prompt", default=None, nargs='?', help="Prompt to run (file name, path, or direct prompt)")
    parser.add_argument("--conversation", default=DEFAULT_CONVERSATION_BASE, help="Base name (detect iteration) or full filename of the conversation file")
    parser.add_argument("--dry", action="store_true", help="Perform a dry run up to inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Specify the model to run.")
    # llm args
    parser.add_argument("--temperature", "-temp", type=float, default=1.0, help="Sampling temperature for the model.")
    parser.add_argument("--top-k", '-topk', type=int, default=None, help="Top-k sampling for the model.")
    parser.add_argument("--top-p", '-topp', type=float, default=1.0, help="Top-p (nucleus) sampling for the model.")
    parser.add_argument("--max-tokens", '-len', type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--frequency", type=float, default=0.0, help="Frequency penalty for the model.")
    parser.add_argument("--presence", type=float, default=0.0, help="Presence penalty for the model.")

    # Parse remaining arguments as tags
    parser.add_argument('-tag', action='append', nargs=2, metavar=('KEY', 'VALUE'), help='Specify custom tags')

    args, unknown = parser.parse_known_args()


    # Process unknown arguments as tags
    inputs = dict()
    for i in range(0, len(unknown)):
        parts = unknown[i].lstrip('-').split('=')
        key = parts[0]
        value = parts[1]
        inputs[key] = value

    # Update inputs with default tags, but don't overwrite existing values
    for key, value in DEFAULT_TAGS.items():
        inputs.setdefault(key, str(value))

    console.print("[bold]Inputs:[/bold]")
    for key, value in inputs.items():
        console.print(f"  [cyan]{key}[/cyan]: {value}")

    # Load conversation
    # ----------------------------------------

    # Determine input file and next iteration
    console.print("[bold]Determining input file and iteration...[/bold]")
    if '.' in args.conversation:
        input_file = args.conversation
        base_name, _ = os.path.splitext(args.conversation)
    else:
        latest_iteration = get_latest_iteration(args.conversation)
        input_file = get_conversation_file(args.conversation, latest_iteration)
        base_name = args.conversation

    # Load the conversation
    conversation = Conversation.from_file(input_file)
    console.print(f"Loaded [green]{len(conversation.messages)}[/green] messages")
    conversation.print_conversation(80)

    if args.prompt is None:
        return

    # Load prompt
    # ----------------------------------------
    prompt = Prompt(args.prompt, inputs)

    # Process the conversation based on the specified method
    # ----------------------------------------
    console.print(f"[bold]Processing conversation using {args.model}...[/bold]")
    refactorer = ConversationRefactorer()
    refactored_convo, local_convo = refactorer(
        prompt,
        conversation,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        dry=args.dry
    )

    if args.dry:
        return

    console.print(f"Processed [green]{len(refactored_convo.messages)}[/green] messages")

    # Save conversation / prompt iteration 
    # ----------------------------------------
    next_iteration = (latest_iteration + 1) if latest_iteration is not None else 1
    if next_iteration is not None:
        output_file = get_conversation_file(base_name, next_iteration)
    else:
        # Use a hash of the prompt content to create a unique filename
        prompt_hash = hashlib.md5(prompt.content.encode()).hexdigest()[:8]
        output_file = f"{base_name}.{prompt_hash}.json"

    if "crispr" in prompt.capabilities:
        refactored_convo.save_to_file(output_file)
        local_convo.save_to_file(f".last_output.json")
        console.print(f"[bold green]Processed conversation written to {output_file}[/bold green]")
    elif "rewrite" in prompt.capabilities:
        last_message_content = refactored_convo.messages[-1].content
        new_prompt_file = f"{base_name}.{next_iteration}.txt"
        with open(new_prompt_file, 'w', encoding='utf-8') as f:
            f.write(last_message_content)
        console.print(f"[bold green]New prompt written to {new_prompt_file}[/bold green]")

    # Log the prompt used for this refactoring
    # ----------------------------------------
    log_file = f"{base_name}.log"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Prompt used:\n{prompt.content}\n\n")
    console.print(f"[bold blue]Refactoring log appended to {log_file}[/bold blue]")



if __name__ == "__main__":
    main()
