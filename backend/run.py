"""Entrypoint — run the voice pipeline."""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if "--test" in sys.argv:
        from app.voice.main_loop import _run_test
        asyncio.run(_run_test())
    elif "--legacy" in sys.argv:
        from app.voice.main_loop import _run
        asyncio.run(_run())
    else:
        from app.voice.pipeline import VoicePipeline
        from app.voice.conversation import ConversationRouter

        async def run():
            pipeline = VoicePipeline(
                conversation=ConversationRouter(),
            )
            await pipeline.run_until_shutdown()

        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
