"""Root CLI launcher for standalone CMDAI."""

from __future__ import annotations

import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CMDAI_ROOT", APP_ROOT)

SRC_PATH = os.path.join(APP_ROOT, "src")
_norm_src = os.path.normcase(os.path.abspath(SRC_PATH))
_norm_root = os.path.normcase(os.path.abspath(APP_ROOT))
sys.path[:] = [
    p for p in sys.path
    if os.path.normcase(os.path.abspath(p or os.getcwd())) not in (_norm_src, _norm_root)
]
if os.path.isdir(SRC_PATH):
    sys.path.insert(0, SRC_PATH)
sys.path.append(APP_ROOT)
if "cmdai" == __name__:
    __path__ = [os.path.join(SRC_PATH, "cmdai")]
    if globals().get("__spec__") is not None:
        __spec__.submodule_search_locations = __path__


def _wants_headless_prompt(argv: list[str]) -> bool:
    for arg in argv:
        if arg in ("-p", "--prompt") or arg.startswith("--prompt="):
            return True
    return False


if __name__ == "__main__":
    if _wants_headless_prompt(sys.argv[1:]):
        import argparse
        from cmdai.core.headless import run_headless

        parser = argparse.ArgumentParser(prog="cmdai", add_help=False)
        parser.add_argument("-p", "--prompt", required=True)
        parser.add_argument("--workdir", default=os.getcwd())
        parser.add_argument("--output-format", default="text")
        parser.add_argument("--continue", dest="continue_session", action="store_true")
        parser.add_argument("--provider", default="")
        parser.add_argument("--model", default="")
        args, _ = parser.parse_known_args()
        sys.exit(
            run_headless(
                prompt=args.prompt,
                workdir=args.workdir,
                output_format=args.output_format,
                continue_session=args.continue_session,
                provider=args.provider,
                model=args.model,
            )
        )
    else:
        from cmdai.cli import main

        main()
