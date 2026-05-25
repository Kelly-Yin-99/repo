import subprocess
import sys
from pathlib import Path

FILES_DIR = Path("/workspace/cogs205b-s26/modules/07-working-with-ai-tools/files")
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

from gemini_simple_api import GeminiSimpleAPI

TASK_DIR = Path(__file__).resolve().parent
TEST_DIR = TASK_DIR / "tests"
TEST_FILE = TEST_DIR / "test_bayes_factor.py"
SOURCE_FILE = TASK_DIR / "bayes_factor.py"
PROMPT_FILE = TASK_DIR / "task.txt"
OUTPUT_FILE = TASK_DIR / "agent_loop_output.txt"

MODEL = "gemma-4-31b-it"
MAX_ATTEMPTS = 5

TEST_FILE.chmod(0o444)

client = GeminiSimpleAPI(
    api_key_file=None,
    model=MODEL,
    working_dir=TASK_DIR,
    protected_directories=[TEST_DIR],
)


def run_tests():
    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", "tests"],
        cwd=TASK_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


prompt_text = PROMPT_FILE.read_text()
OUTPUT_FILE.write_text("")

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\nAttempt {attempt}")

    with OUTPUT_FILE.open("a") as f:
        f.write(f"\nAttempt {attempt}\n")

    try:
        files, notes = client.prompt(
            prompt=prompt_text,
            attachments=[SOURCE_FILE, TEST_FILE],
            verbose=True,
        )

        with OUTPUT_FILE.open("a") as f:
            f.write("Model wrote files:\n")
            for path in files:
                f.write(f"+ {path}\n")
            if notes:
                f.write(f"Notes: {notes}\n")

    except Exception as e:
        print("Model/API error:")
        print(e)

        with OUTPUT_FILE.open("a") as f:
            f.write("Model/API error:\n")
            f.write(str(e))
            f.write("\n")

    code, output = run_tests()
    print(output)

    with OUTPUT_FILE.open("a") as f:
        f.write("\nTest output:\n")
        f.write(output)
        f.write("\n")

    if code == 0:
        print("Tests passed.")
        break

    prompt_text += (
        "\n\nThe previous attempt failed.\n"
        "Here is the test output or model/API error:\n"
        f"{output}\n"
        "Return only valid structured JSON and fix bayes_factor.py."
    )
else:
    print("Stopped after maximum attempts.")
