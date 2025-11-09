from __future__ import annotations

import argparse
import json
from typing import List

from .config import Config, ensure_dirs
from .pipeline_ingest import IngestPipeline
from .pipeline_retrieve import Peer, RetrievePipeline
from .storage import JsonStore


def cmd_init(args: argparse.Namespace) -> None:
    cfg = Config()
    ensure_dirs(cfg)
    store = JsonStore(cfg)
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = Config()
    pipe = IngestPipeline(cfg)
    # load user profile
    if args.profile and args.profile.endswith(".json"):
        with open(args.profile, "r", encoding="utf-8") as f:
            p = json.load(f)
            profile_text = p.get("profile_text", "")
    else:
        profile_text = args.profile or ""
    pipe.ensure_user(args.user_id, profile_text)
    # load dialog text
    if args.dialog.endswith(".txt"):
        with open(args.dialog, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raw_text = args.dialog
    item = pipe.ingest_dialog(args.user_id, raw_text)
    if item is None:
        print(
            json.dumps(
                {"include": False, "reason": "QC rejected"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            {"include": True, "item": item.to_dict()}, ensure_ascii=False, indent=2
        )
    )


def _load_peers(path: str) -> List[Peer]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    peers: List[Peer] = []
    for p in arr:
        peers.append(Peer(user_id=p["user_id"], profile_text=p["profile_text"]))
    return peers


def cmd_retrieve(args: argparse.Namespace) -> None:
    cfg = Config()
    ret = RetrievePipeline(cfg)
    store = JsonStore(cfg)
    user = store.get_user(args.user_id)
    if user is None:
        print(
            json.dumps({"error": f"user {args.user_id} not found"}, ensure_ascii=False)
        )
        return
    peers = _load_peers(args.peers) if args.peers else []
    res = ret.retrieve(user, args.task or "", peers, top_k=args.top_k)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_prompt(args: argparse.Namespace) -> None:
    cfg = Config()
    ret = RetrievePipeline(cfg)
    store = JsonStore(cfg)
    user = store.get_user(args.user_id)
    if user is None:
        print("user not found")
        return
    peers = _load_peers(args.peers) if args.peers else []
    res = ret.retrieve(user, args.task or "", peers, top_k=args.top_k)
    prompt = ret.build_prompt_blocks(res["items"]) if res.get("items") else ""
    print(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(prog="sharememory_user")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.set_defaults(func=cmd_init)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--user_id", required=True)
    p_ing.add_argument(
        "--profile", required=True, help="path to user profile json or raw text"
    )
    p_ing.add_argument(
        "--dialog", required=True, help="path to dialog .txt or raw text"
    )
    p_ing.set_defaults(func=cmd_ingest)

    p_ret = sub.add_parser("retrieve")
    p_ret.add_argument("--user_id", required=True)
    p_ret.add_argument("--task", required=False, default="")
    p_ret.add_argument(
        "--peers", required=False, default="", help="path to peers json array"
    )
    p_ret.add_argument("--top_k", type=int, default=5)
    p_ret.set_defaults(func=cmd_retrieve)

    p_pr = sub.add_parser("prompt")
    p_pr.add_argument("--user_id", required=True)
    p_pr.add_argument("--task", required=False, default="")
    p_pr.add_argument("--peers", required=False, default="")
    p_pr.add_argument("--top_k", type=int, default=5)
    p_pr.set_defaults(func=cmd_prompt)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
