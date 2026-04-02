import asyncio

from dotenv import load_dotenv

load_dotenv()

import notebook_lm


async def main():
    await notebook_lm.startup()

    print("\nNotebookLM ready. Ask questions about your notes.")
    print("Tip: prefix with 'brief:' for a concise answer.")
    print("Type 'new' to start a fresh conversation. Type 'quit' to exit.\n")

    thread_id: str | None = None

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if query.lower() == "new":
            thread_id = None
            print("\n[New conversation started]\n")
            continue

        result = await notebook_lm.ask(query, thread_id=thread_id)

        if thread_id is None:
            thread_id = result.thread_id
            print(f"[Thread: {thread_id}]\n")

        print(f"\nAgent: {result.answer}\n")

        if result.summary:
            print(f"[Summary]\n{result.summary}\n")


if __name__ == "__main__":
    asyncio.run(main())
