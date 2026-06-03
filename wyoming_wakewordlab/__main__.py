import argparse
import asyncio
import logging
from functools import partial

import wakewordlab
from wyoming.info import Attribution, Info, WakeModel, WakeProgram
from wyoming.server import AsyncServer

from .handler import WakeWordHandler

_LOGGER = logging.getLogger(__name__)


def _build_info(model_names: list[str], sessions: dict) -> Info:
    models = []
    for name in model_names:
        session = sessions[name]
        phrase = session.info.wake_word
        models.append(
            WakeModel(
                name=name,
                phrase=phrase,
                attribution=Attribution(
                    name="wakewordlab",
                    url="https://github.com/ubermorgenland/wakewordlab",
                ),
                installed=True,
                description=f"Wake word: {phrase}",
                version="1.0.0",
            )
        )
    return Info(wake=[WakeProgram(name="wakewordlab", models=models)])


async def run(args: argparse.Namespace) -> None:
    sessions = {}
    for slug in args.models:
        slug = slug.strip()
        _LOGGER.info("Loading model: %s", slug)
        wakewordlab.download(slug)
        detector = wakewordlab.WakewordDetector(slug, vad=False)
        sessions[slug] = detector._session

    wyoming_info = _build_info(args.models, sessions)
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Listening on %s", args.uri)
    await server.run(partial(WakeWordHandler, wyoming_info, args, sessions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming wake word server — wakewordlab")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10400", help="Server URI")
    parser.add_argument("--models", nargs="+", default=["hey_jarvis"], help="Model slugs to load")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--refractory-seconds", type=float, default=2.0, help="Cooldown between detections")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
