"""Play a game against BitterBOT from the terminal.

    python play.py                 # you play White, engine gets 2s/move
    python play.py --color black    # you play Black
    python play.py --think-ms 5000  # give the engine more time per move

Enter moves in SAN ("Nf3", "exd5", "O-O") or UCI ("g1f3", "e4d5", "e1g1").
"""
import argparse

import chess

import agent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--color", choices=["white", "black"], default="white",
                     help="the side you play (default: white)")
    ap.add_argument("--think-ms", type=int, default=2000,
                     help="time budget per engine move, in milliseconds (default: 2000)")
    args = ap.parse_args()
    human_is_white = args.color == "white"

    board = chess.Board()
    print(board, "\n")

    while not board.is_game_over():
        human_to_move = board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK
        if human_to_move:
            move = None
            while move is None:
                text = input("your move: ").strip()
                try:
                    move = board.parse_san(text)
                except ValueError:
                    try:
                        move = board.parse_uci(text)
                    except ValueError:
                        print("  not a legal move, try again")
            board.push(move)
        else:
            uci = agent.get_move(board.fen(), args.think_ms)
            move = chess.Move.from_uci(uci)
            print(f"BitterBOT plays: {board.san(move)}")
            board.push(move)
        print(board, "\n")

    print("Game over:", board.result(), "-", board.outcome().termination.name)


if __name__ == "__main__":
    main()
