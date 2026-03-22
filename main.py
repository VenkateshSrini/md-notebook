import asyncio

from dotenv import load_dotenv

load_dotenv()

import notebook_lm


async def main():
    await notebook_lm.startup()

    print("\nNotebookLM ready. Ask questions about your notes.")
    print("Tip: Start with 'brief:' for a short answer, or just ask for detailed responses.")
    print("Type 'quit' to exit.\n")

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

        response = await notebook_lm.ask(query)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
