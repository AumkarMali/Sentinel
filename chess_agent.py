"""
Chess Engine Module - Headless chess agent (no UI).

Provides ChessEngine: YOLO piece detection + Stockfish analysis + move execution.
The main GUI drives this engine and handles all display/logging.
"""
import os
import time
import numpy as np
import pyautogui
import chess
from PIL import Image, ImageDraw
from ultralytics import YOLO
from stockfish import Stockfish

# ── paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chess_model.pt")
STOCKFISH_PATH = os.path.join(BASE_DIR, "stockfish.exe")

# ── game-over keywords (lowercase) ──
GAME_OVER_KEYWORDS = [
    "won", "wins", "lost", "lose", "aborted", "draw", "drawn",
    "stalemate", "checkmate", "resigned", "timeout", "victory",
    "defeat", "game over", "1-0", "0-1", "1/2", "rematch",
]

# ── YOLO class -> python-chess piece mapping ──
YOLO_TO_PIECE = {
    "white_king":   chess.Piece(chess.KING,   chess.WHITE),
    "white_queen":  chess.Piece(chess.QUEEN,  chess.WHITE),
    "white_rook":   chess.Piece(chess.ROOK,   chess.WHITE),
    "white_bishop": chess.Piece(chess.BISHOP, chess.WHITE),
    "white_knight": chess.Piece(chess.KNIGHT, chess.WHITE),
    "white_pawn":   chess.Piece(chess.PAWN,   chess.WHITE),
    "black_king":   chess.Piece(chess.KING,   chess.BLACK),
    "black_queen":  chess.Piece(chess.QUEEN,  chess.BLACK),
    "black_rook":   chess.Piece(chess.ROOK,   chess.BLACK),
    "black_bishop": chess.Piece(chess.BISHOP, chess.BLACK),
    "black_knight": chess.Piece(chess.KNIGHT, chess.BLACK),
    "black_pawn":   chess.Piece(chess.PAWN,   chess.BLACK),
}

# Keywords that trigger chess mode
CHESS_KEYWORDS = [
    "chess", "play chess", "chess game", "chess agent",
    "chess match", "stockfish", "lichess", "chess.com",
]


# ======================================================================
#  Board-state helpers
# ======================================================================

def _filter_board_pieces(pieces):
    """Remove YOLO detections that are likely sidebar/UI icons, not board pieces.

    Chess.com and similar sites show small piece icons in move history and captured
    pieces panels.  Board pieces are all roughly the same (large) size; UI icons are
    much smaller.  We keep only detections whose bounding-box area is within 3× of
    the median, which reliably removes tiny icons while keeping all board pieces.
    """
    if len(pieces) < 4:
        return pieces
    areas = [(p["x2"] - p["x1"]) * (p["y2"] - p["y1"]) for p in pieces]
    med = float(np.median(areas))
    if med <= 0:
        return pieces
    return [p for p, a in zip(pieces, areas) if 0.25 * med <= a <= 4.0 * med]


def _filter_by_board_box(pieces, board_box):
    """Spatial filter: remove pieces whose centres fall outside the board box.

    This is the most reliable way to eliminate sidebar/move-history icons that
    survive the area filter because they happen to be similar in size to pawns.
    Lichess places its move-history panel immediately to the right of the board,
    so we allow only a 0.6-square margin beyond each edge — wide enough to catch
    all board-piece centres (which are at most 0.5 squares from the edge) but
    narrow enough to reject adjacent sidebar icons.
    """
    if not board_box or not pieces:
        return pieces
    bx1, by1 = board_box["x1"], board_box["y1"]
    bx2, by2 = board_box["x2"], board_box["y2"]
    sq_w = (bx2 - bx1) / 8
    sq_h = (by2 - by1) / 8
    mx, my = sq_w * 0.6, sq_h * 0.6   # ~0.6 sq margin — keeps edge pieces, rejects sidebar
    return [p for p in pieces
            if bx1 - mx <= p["cx"] <= bx2 + mx
            and by1 - my <= p["cy"] <= by2 + my]


def _estimate_sq_size(coords):
    """Estimate chess square size (px) from 1-D piece-centre coordinates.

    Uses the smallest gap between adjacent piece centres that is at least
    MIN_SQ (30 px) — this is the single-square width.  The GCD approach is
    fragile when sidebar pieces at mixed y-positions are included; the min-gap
    approach with a meaningful lower bound is far more robust.
    """
    MIN_SQ = 30  # minimum plausible square size in pixels

    uniq = sorted(set(round(c / 2) * 2 for c in coords))  # 2-px quantisation
    if len(uniq) < 2:
        return None
    diffs = [int(uniq[i + 1] - uniq[i]) for i in range(len(uniq) - 1)
             if uniq[i + 1] - uniq[i] >= MIN_SQ]
    if not diffs:
        return None
    return float(min(diffs))


def _board_box_from_pieces(pieces):
    """Compute a reliable board bounding box from piece positions.

    Uses inter-piece distances to estimate square size, then places board edges
    exactly 0.5 squares from the outermost piece centres.  Much more accurate
    than a fixed-percentage padding, especially for opening positions where
    ranks 3-6 are empty.
    """
    xs = [p["cx"] for p in pieces]
    ys = [p["cy"] for p in pieces]

    sq_w = _estimate_sq_size(xs)
    sq_h = _estimate_sq_size(ys)

    # Fallback: assume pieces span 7 of the 8 columns/rows (all corners occupied)
    x_range = max(xs) - min(xs) if len(xs) > 1 else 0
    y_range = max(ys) - min(ys) if len(ys) > 1 else 0
    if sq_w is None or sq_w < 4:
        sq_w = x_range / 7 if x_range > 0 else 40
    if sq_h is None or sq_h < 4:
        sq_h = y_range / 7 if y_range > 0 else 40

    bx1 = min(xs) - 0.5 * sq_w
    by1 = min(ys) - 0.5 * sq_h
    return {
        "x1": bx1, "y1": by1,
        "x2": bx1 + 8 * sq_w, "y2": by1 + 8 * sq_h,
        "conf": 0.0,
    }


def _collision_rate(pieces, board_box):
    """Return fraction of pieces that share a square with another piece (0.0–1.0)."""
    if not pieces or not board_box:
        return 0.0
    bx1, by1 = board_box["x1"], board_box["y1"]
    sq_w = (board_box["x2"] - bx1) / 8
    sq_h = (board_box["y2"] - by1) / 8
    seen: dict = {}
    collisions = 0
    for p in pieces:
        col = max(0, min(7, int((p["cx"] - bx1) / sq_w)))
        row = max(0, min(7, int((p["cy"] - by1) / sq_h)))
        key = (col, row)
        if key in seen:
            collisions += 1
        seen[key] = True
    return collisions / len(pieces)


def detections_to_board(pieces, board_box):
    """Map YOLO-detected pieces onto an 8x8 chess.Board."""
    if board_box:
        bx1, by1 = board_box["x1"], board_box["y1"]
        bx2, by2 = board_box["x2"], board_box["y2"]
    else:
        box = _board_box_from_pieces(pieces)
        bx1, by1 = box["x1"], box["y1"]
        bx2, by2 = box["x2"], box["y2"]

    sq_w = (bx2 - bx1) / 8
    sq_h = (by2 - by1) / 8

    white_y, white_n = 0, 0
    black_y, black_n = 0, 0
    for p in pieces:
        if p["label"].startswith("white"):
            white_y += p["cy"]; white_n += 1
        elif p["label"].startswith("black"):
            black_y += p["cy"]; black_n += 1

    orientation = ("white"
                   if (white_y / max(white_n, 1)) > (black_y / max(black_n, 1))
                   else "black")

    board = chess.Board(fen=None)
    board.clear()

    for p in pieces:
        label = p["label"]
        if label not in YOLO_TO_PIECE:
            continue
        piece = YOLO_TO_PIECE[label]
        col = max(0, min(7, int((p["cx"] - bx1) / sq_w)))
        row = max(0, min(7, int((p["cy"] - by1) / sq_h)))
        if orientation == "white":
            sq = chess.square(col, 7 - row)
        else:
            sq = chess.square(7 - col, row)
        board.set_piece_at(sq, piece)

    # Infer castling rights
    board.castling_rights = chess.BB_EMPTY
    if board.king(chess.WHITE) == chess.E1:
        if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            board.castling_rights |= chess.BB_H1
        if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            board.castling_rights |= chess.BB_A1
    if board.king(chess.BLACK) == chess.E8:
        if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            board.castling_rights |= chess.BB_H8
        if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            board.castling_rights |= chess.BB_A8

    return board, orientation


def board_ascii(board, orientation):
    """Pretty-print the board from the given orientation."""
    lines = []
    if orientation == "white":
        for rank in range(7, -1, -1):
            row = f" {rank+1}  "
            for f in range(8):
                p = board.piece_at(chess.square(f, rank))
                row += f" {p.symbol() if p else '.'}"
            lines.append(row)
        lines.append("     a b c d e f g h")
    else:
        for rank in range(8):
            row = f" {rank+1}  "
            for f in range(7, -1, -1):
                p = board.piece_at(chess.square(f, rank))
                row += f" {p.symbol() if p else '.'}"
            lines.append(row)
        lines.append("     h g f e d c b a")
    return "\n".join(lines)


def square_to_screen(sq_name, board_box, orientation):
    """Convert chess square name (e.g. 'e2') to screen pixel center."""
    bx1, by1 = board_box["x1"], board_box["y1"]
    sq_w = (board_box["x2"] - bx1) / 8
    sq_h = (board_box["y2"] - by1) / 8
    fi = ord(sq_name[0]) - ord('a')
    ri = int(sq_name[1]) - 1
    if orientation == "white":
        px = bx1 + fi * sq_w + sq_w / 2
        py = by1 + (7 - ri) * sq_h + sq_h / 2
    else:
        px = bx1 + (7 - fi) * sq_w + sq_w / 2
        py = by1 + ri * sq_h + sq_h / 2
    return int(px), int(py)


def is_chess_task(text):
    """Return True if the task text looks like a chess request."""
    t = text.lower()
    return any(kw in t for kw in CHESS_KEYWORDS)


# ======================================================================
#  Chess Engine (headless)
# ======================================================================

class ChessEngine:
    """YOLO + Stockfish chess engine with no UI.

    Parameters
    ----------
    log_fn : callable(msg, tag)
        Function the engine calls to emit log messages.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg, tag="": None)
        self.model = None
        self.engine = None
        self.ready = False

        # per-game state
        self._last_fen = None
        self._last_good_board_box = None   # cached board box from last clean detection
        self._move_count = 0
        self._cycle_count = 0
        self._unchanged_count = 0
        self._no_move_count = 0
        self._ocr_reader = None

    def log(self, msg, tag=""):
        self._log(msg, tag)

    # ------------------------------------------------------------------
    #  Loading
    # ------------------------------------------------------------------
    def load_models(self):
        """Load YOLO and Stockfish. Returns True on success."""
        ok = True
        try:
            t0 = time.time()
            self.model = YOLO(MODEL_PATH)
            self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
            self.log(f"  YOLO chess model loaded ({time.time()-t0:.1f}s)", "info")
        except Exception as e:
            self.log(f"  YOLO FAILED: {e}", "error")
            ok = False

        try:
            t0 = time.time()
            if not os.path.isfile(STOCKFISH_PATH):
                raise FileNotFoundError(f"stockfish.exe not found")
            self.engine = Stockfish(
                path=STOCKFISH_PATH, depth=18,
                parameters={"Threads": 2, "Hash": 128})
            self.log(f"  Stockfish loaded ({time.time()-t0:.1f}s)", "info")
        except Exception as e:
            self.log(f"  STOCKFISH FAILED: {e}", "error")
            ok = False

        self.ready = ok
        return ok

    # ------------------------------------------------------------------
    #  State
    # ------------------------------------------------------------------
    def reset(self):
        """Reset per-game state for a new game."""
        self._last_fen = None
        self._last_good_board_box = None
        self._move_count = 0
        self._cycle_count = 0
        self._unchanged_count = 0
        self._no_move_count = 0

    @property
    def move_count(self):
        return self._move_count

    @property
    def cycle_count(self):
        return self._cycle_count

    # ------------------------------------------------------------------
    #  Main analysis
    # ------------------------------------------------------------------
    def analyze(self, screenshot, conf, depth, turn, force_move=False):
        """Run one full cycle on a screenshot.

        Returns a dict:
            status   : "move" | "waiting" | "game_over" | "no_board" | "error"
            best_move: e.g. "e2e4" (only when status=="move")
            eval     : e.g. "+0.35" or "M3"
            fen      : full FEN string
            pieces   : list of piece dicts
            board_box: board bounding box dict or None
            orientation: "white" or "black"
            annotated: PIL Image with detection overlay
            top_moves: list of top moves from Stockfish
        """
        self._cycle_count += 1

        res = dict(status="error", best_move=None, eval=None, fen=None,
                   pieces=[], board_box=None, orientation="white",
                   annotated=None, top_moves=[], message="")

        # ── YOLO detection ──
        img_arr = np.array(screenshot.convert("RGB"))
        t0 = time.time()
        results = self.model.predict(img_arr, conf=conf, verbose=False)
        dt = time.time() - t0

        pieces, board_box = [], None
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                c = float(box.conf[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if label == "board":
                    board_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c}
                else:
                    pieces.append({"label": label, "confidence": c,
                                   "cx": cx, "cy": cy,
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        res["pieces"] = pieces

        if not pieces:
            self.log(f"  No pieces found ({dt:.2f}s)", "dim")
            res["status"] = "no_board"
            return res

        # ── Step 1: area filter — remove obvious tiny sidebar icons ──
        filtered = _filter_board_pieces(pieces)
        if len(filtered) >= 4:
            if len(filtered) < len(pieces):
                self.log(f"  Area filter: removed {len(pieces) - len(filtered)} small detections "
                         f"({len(filtered)} remain)", "dim")
            pieces = filtered

        # ── Step 2: board box (YOLO if available, else use cached or estimate) ──
        if board_box is None:
            if self._last_good_board_box is not None:
                board_box = self._last_good_board_box
                self.log(f"  Board box from cache (YOLO missed label): "
                         f"({board_box['x1']:.0f},{board_box['y1']:.0f})-"
                         f"({board_box['x2']:.0f},{board_box['y2']:.0f})", "dim")
            else:
                board_box = _board_box_from_pieces(pieces)
                self.log(f"  Board box estimated from pieces: "
                         f"({board_box['x1']:.0f},{board_box['y1']:.0f})-"
                         f"({board_box['x2']:.0f},{board_box['y2']:.0f})", "dim")

        # ── Step 3: SPATIAL filter — remove pieces outside the board box ──
        # This eliminates sidebar / move-history icons whose bounding-box area is
        # similar to real pieces (area filter alone misses them).  Running this
        # AFTER we have the board_box prevents them from corrupting sq-size estimates.
        spatial = _filter_by_board_box(pieces, board_box)
        if len(spatial) >= 4:
            if len(spatial) < len(pieces):
                self.log(f"  Spatial filter: removed {len(pieces) - len(spatial)} pieces "
                         f"outside board box ({len(spatial)} remain)", "dim")
            pieces = spatial

        res["pieces"] = pieces

        # ── Step 4: collision check — only recompute board_box when it is BETTER ──
        cr = _collision_rate(pieces, board_box)
        if cr > 0.25:
            self.log(f"  Collision rate {cr:.0%} — trying to recompute board box", "dim")
            new_bb = _board_box_from_pieces(pieces)
            cr2 = _collision_rate(pieces, new_bb)
            if cr2 < cr:
                self.log(f"  Board box improved: collision {cr:.0%} → {cr2:.0%}", "dim")
                board_box = new_bb
            else:
                self.log(f"  Recomputed board box worse ({cr2:.0%} ≥ {cr:.0%}) — keeping original", "dim")

        # Cache this board box if it's clean (low collisions, reasonable piece count).
        # Future scans use it as fallback when YOLO misses the board label.
        if _collision_rate(pieces, board_box) < 0.15 and len(pieces) >= 16:
            self._last_good_board_box = board_box

        res["board_box"] = board_box

        # ── Map to board ──
        board, orientation = detections_to_board(pieces, board_box)
        fen_pieces = board.fen().split(" ")[0]
        our_side = chess.WHITE if turn == "white" else chess.BLACK
        # After a board change the opponent moved → it's our turn; otherwise keep our_side
        if self._last_fen is not None and fen_pieces != self._last_fen:
            board.turn = our_side
        else:
            board.turn = our_side  # default to our side so Stockfish always evaluates for us
        fen = board.fen()

        res["fen"] = fen
        res["orientation"] = orientation

        # ── Board unchanged? ──
        if self._last_fen == fen_pieces and not force_move:
            self._unchanged_count += 1
            self.log(f"  Board unchanged, waiting... ({dt:.2f}s) (#{self._unchanged_count})", "dim")

            # Check for game-over text only every 4th unchanged scan to avoid
            # spending 3–4s on OCR every cycle while waiting for the opponent.
            if self._unchanged_count % 4 == 0:
                self.log("  Checking for game-over text...", "dim")
                kw = self._check_game_over(screenshot, board_box)
                if kw:
                    self.log(f"  GAME OVER detected: \"{kw}\"", "action")
                    res["status"] = "game_over"
                    res["message"] = kw
                    return res

            res["status"] = "waiting"
            res["annotated"] = self._draw(screenshot, pieces, board_box,
                                          orientation=orientation)
            return res

        # ── Board changed ──
        self._unchanged_count = 0
        self.log(f"  {len(pieces)} pieces in {dt:.2f}s", "info")

        counts = {}
        for p in pieces:
            counts[p["label"]] = counts.get(p["label"], 0) + 1
        cstr = ", ".join(f"{n.split('_')[1][0].upper()}{c}"
                         for n, c in sorted(counts.items()) if n != "board")
        self.log(f"  Pieces: {cstr}", "piece")
        self.log(f"  FEN: {fen}", "action")

        for line in board_ascii(board, orientation).split("\n"):
            self.log(f"  {line}", "board")

        res["annotated"] = self._draw(screenshot, pieces, board_box,
                                      orientation=orientation)

        # ── Validate kings ──
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            missing = []
            if board.king(chess.WHITE) is None:
                missing.append("white king")
            if board.king(chess.BLACK) is None:
                missing.append("black king")
            self._unchanged_count += 1
            self.log(f"  SKIP: Missing {', '.join(missing)} "
                     f"(#{self._unchanged_count})", "error")

            # Only check for game-over text every 3rd consecutive failure to avoid
            # slow OCR loading on every scan.
            if self._unchanged_count % 3 == 0:
                self.log("  Checking for game-over text...", "dim")
                kw = self._check_game_over(screenshot, board_box)
                if kw:
                    self.log(f"  *** GAME OVER: \"{kw}\" ***", "action")
                    res["status"] = "game_over"
                    res["message"] = kw
                    return res

            # Only declare board obstructed after many consecutive failures.
            # 6 was far too low: a 3-scan opponent-wait raises the counter to 3,
            # and then just 3 animation frames with missing kings would trigger it.
            # 20 gives the game enough headroom to survive mid-move animations.
            if self._unchanged_count >= 20:
                self.log("  Board detection failed 20x in a row — game likely over", "action")
                res["status"] = "game_over"
                res["message"] = "board obstructed"
                return res

            res["status"] = "error"
            return res

        # ── Stockfish ──
        try:
            t0 = time.time()
            self.engine.set_depth(depth)
            self.engine.set_fen_position(fen)
            best = self.engine.get_best_move()
            top = self.engine.get_top_moves(3)
            ev = self.engine.get_evaluation()
            et = time.time() - t0

            ev_str = (f"{ev['value']/100:+.2f}" if ev["type"] == "cp"
                      else f"M{ev['value']}")
            res["eval"] = ev_str
            res["top_moves"] = top
        except Exception as e:
            self.log(f"  Stockfish error: {e}", "error")
            self._restart_engine()
            res["status"] = "error"
            return res

        if not best:
            self._no_move_count += 1
            if self._no_move_count >= 6:
                self.log(f"  No legal move 6× in a row — declaring game over", "warning")
                res["status"] = "game_over"
                res["message"] = "no legal moves"
            else:
                self.log(f"  No legal move ({self._no_move_count}/6) — detection may be glitchy; retrying", "warning")
                res["status"] = "error"
            return res
        self._no_move_count = 0  # reset on success

        # ── Success: our turn, play the move ──
        res["best_move"] = best
        res["status"] = "move"

        fsq, tsq = best[:2], best[2:4]
        promo = f"={best[4:].upper()}" if len(best) > 4 else ""
        self.log(f"  >> BEST: {fsq} -> {tsq}{promo}  "
                 f"eval={ev_str}  ({et:.2f}s)", "move")

        for i, m in enumerate(top[:3], 1):
            mv = m["Move"]
            sc = (f"M{m['Mate']}" if m["Mate"] is not None
                  else f"{m['Centipawn']/100:+.1f}")
            self.log(f"     {i}. {mv[:2]}->{mv[2:4]}  ({sc})", "dim")

        res["annotated"] = self._draw(screenshot, pieces, board_box,
                                      best_move=best, orientation=orientation)
        return res

    # ------------------------------------------------------------------
    #  Move execution
    # ------------------------------------------------------------------
    def execute_move(self, best_move, board_box, orientation, click_delay=0.5):
        """Drag the piece on screen from source to destination square.

        Drag (moveTo → mouseDown → moveTo → mouseUp) is far more reliable than
        two separate clicks because:
        1.  The initial moveTo hovers over the source square, which causes the
            browser window to receive mouse-focus automatically.
        2.  The entire operation is one continuous mouse gesture — Chrome/lichess
            cannot miss it the way it can miss two independent click() calls when
            the window hasn't fully grabbed focus yet.
        3.  Both chess.com and lichess accept drag-to-move natively.
        """
        pyautogui.FAILSAFE = False

        fsq, tsq = best_move[:2], best_move[2:4]
        promo = best_move[4:] if len(best_move) > 4 else None

        fx, fy = square_to_screen(fsq, board_box, orientation)
        tx, ty = square_to_screen(tsq, board_box, orientation)

        bb = board_box
        sq_w = (bb["x2"] - bb["x1"]) / 8
        sq_h = (bb["y2"] - bb["y1"]) / 8
        self.log(
            f"  Drag: {fsq}({fx},{fy}) → {tsq}({tx},{ty})  "
            f"sq={sq_w:.0f}x{sq_h:.0f}px  "
            f"board=({bb['x1']:.0f},{bb['y1']:.0f})-({bb['x2']:.0f},{bb['y2']:.0f})"
            f"  orient={orientation}", "dim"
        )

        # Sanity-check: square size must be in a plausible range.
        if not (20 <= sq_w <= 400 and 20 <= sq_h <= 400):
            self.log(
                f"  WARNING: square size {sq_w:.0f}x{sq_h:.0f}px out of range — skipping",
                "error"
            )
            return

        # 1. Glide cursor to the source square — hovering gives the browser focus.
        pyautogui.moveTo(fx, fy, duration=0.25)
        time.sleep(0.25)   # let the browser register the hover and take focus

        # 2. Press mouse button (pick up the piece).
        pyautogui.mouseDown(button="left")
        time.sleep(0.15)

        # 3. Drag to destination (move the piece).
        pyautogui.moveTo(tx, ty, duration=0.35)
        time.sleep(0.15)

        # 4. Release (place the piece).
        pyautogui.mouseUp(button="left")
        time.sleep(0.2)

        # Handle promotion: click the promotion square a second time if needed.
        if promo:
            time.sleep(0.4)
            pyautogui.click(tx, ty)

        self._move_count += 1

    # ------------------------------------------------------------------
    #  Post-move re-scan
    # ------------------------------------------------------------------
    def capture_post_move_fen(self, screenshot, conf):
        """Re-detect the board after our move to track opponent changes.

        Uses the same filtering and board-box logic as analyze() so the saved
        FEN is computed on the same piece set and coordinate system, preventing
        spurious "board changed" signals on the next scan.
        """
        try:
            img_arr = np.array(screenshot.convert("RGB"))
            results = self.model.predict(img_arr, conf=conf, verbose=False)

            pieces, board_box = [], None
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    c = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if label == "board":
                        board_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c}
                    else:
                        pieces.append({"label": label, "confidence": c,
                                       "cx": cx, "cy": cy,
                                       "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            if pieces:
                # Mirror analyze()'s 4-step filtering pipeline exactly
                filtered = _filter_board_pieces(pieces)
                if len(filtered) >= 4:
                    pieces = filtered

                if board_box is None:
                    if self._last_good_board_box is not None:
                        board_box = self._last_good_board_box
                    else:
                        board_box = _board_box_from_pieces(pieces)

                spatial = _filter_by_board_box(pieces, board_box)
                if len(spatial) >= 4:
                    pieces = spatial

                cr = _collision_rate(pieces, board_box)
                if cr > 0.25:
                    new_bb = _board_box_from_pieces(pieces)
                    if _collision_rate(pieces, new_bb) < cr:
                        board_box = new_bb

                # Sanity check: if too few pieces remain after filtering, the board box
                # is almost certainly wrong (animation frame, YOLO miss, etc.).
                # Don't save a corrupt FEN — clear it so the next scan re-analyses cleanly.
                if len(pieces) < 12:
                    self._last_fen = None
                    self.log(f"  Post-move scan: only {len(pieces)} pieces — board box unreliable, "
                             "not saving FEN", "dim")
                    return

                board, _ = detections_to_board(pieces, board_box)
                self._last_fen = board.fen().split(" ")[0]
                self.log(f"  Saved post-move board state ({len(pieces)} pieces)", "dim")
            else:
                self._last_fen = None
        except Exception:
            self._last_fen = None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _restart_engine(self):
        try:
            self.engine = Stockfish(
                path=STOCKFISH_PATH, depth=18,
                parameters={"Threads": 2, "Hash": 128})
            self.log("  Engine restarted", "info")
        except Exception:
            self.log("  Engine restart FAILED", "error")

    def _check_game_over(self, screenshot, board_box):
        """OCR the board area for game-over keywords. Returns keyword or None."""
        try:
            if self._ocr_reader is None:
                self.log("  Loading OCR reader (first time)...", "dim")
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.log("  OCR reader ready", "dim")

            if board_box:
                pad = 120
                left = max(0, board_box["x1"] - pad)
                top = max(0, board_box["y1"] - pad)
                right = min(screenshot.width, board_box["x2"] + pad)
                bottom = min(screenshot.height, board_box["y2"] + pad)
            else:
                w, h = screenshot.width, screenshot.height
                left, top = int(w * 0.25), int(h * 0.15)
                right, bottom = int(w * 0.75), int(h * 0.85)

            crop = screenshot.crop((left, top, right, bottom))
            crop_arr = np.array(crop.convert("RGB"))
            results = self._ocr_reader.readtext(crop_arr, detail=0)
            text = " ".join(results).lower()
            self.log(f"  OCR text: \"{text[:120]}\"", "dim")

            for kw in GAME_OVER_KEYWORDS:
                if kw in text:
                    return kw

            self.log("  No game-over keyword matched", "dim")
        except Exception as e:
            self.log(f"  OCR error: {e}", "error")
        return None

    def _draw(self, screenshot, pieces, board_box,
              best_move=None, orientation="white"):
        """Return annotated screenshot with detections and move highlights."""
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)

        if board_box:
            draw.rectangle([board_box["x1"], board_box["y1"],
                            board_box["x2"], board_box["y2"]],
                           outline="#FFD700", width=3)

        pc = {"king": "#FF1493", "queen": "#FFD700", "rook": "#FF8800",
              "bishop": "#AA00FF", "knight": "#00AAFF", "pawn": "#AAAAAA"}

        for p in pieces:
            color = "#00FF00"
            for key, c in pc.items():
                if key in p["label"]:
                    color = c
                    break
            if "black" in p["label"]:
                color = "#" + "".join(
                    format(max(0, int(color[i:i+2], 16) - 60), "02x")
                    for i in (1, 3, 5))
            draw.rectangle([p["x1"], p["y1"], p["x2"], p["y2"]],
                           outline=color, width=2)
            lbl = f"{p['label'].split('_')[1]} {p['confidence']:.0%}"
            tb = draw.textbbox((p["x1"], p["y1"] - 14), lbl)
            draw.rectangle(tb, fill=color)
            draw.text((p["x1"], p["y1"] - 14), lbl, fill="black")

        if best_move and board_box:
            bx1, by1 = board_box["x1"], board_box["y1"]
            sw = (board_box["x2"] - bx1) / 8
            sh = (board_box["y2"] - by1) / 8
            for sq_name, sq_color in [(best_move[:2], "#00FF00"),
                                      (best_move[2:4], "#FFFF00")]:
                fi = ord(sq_name[0]) - ord('a')
                ri = int(sq_name[1]) - 1
                if orientation == "white":
                    px, py = bx1 + fi * sw, by1 + (7 - ri) * sh
                else:
                    px, py = bx1 + (7 - fi) * sw, by1 + ri * sh
                draw.rectangle([int(px), int(py),
                                int(px + sw), int(py + sh)],
                               outline=sq_color, width=4)
        return img
