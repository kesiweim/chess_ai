import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
import torch

from model_cnn import ChessCNN
from search_engine import policy_order
from neural_search_v4 import NeuralSearchV4
from neural_inference_v3 import TracedEvaluator


def parse_move(board, text):
    try:
        move = board.parse_san(text.replace('0-0', 'O-O'))
    except ValueError:
        move = board.parse_uci(text)
    if move not in board.legal_moves:
        raise ValueError('非法走法')
    return move


def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(eval='neural')
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300 or not 1 <= args.depth <= 10:
        parser.error('seconds 必须在 (0,300]，depth 在 1～10')
    root = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    device = torch.device('cpu')
    model_path = args.model
    if args.eval == 'neural':
        if model_path is None:
            candidates = sorted((root / 'value_full_runs').glob('*/value_full_epoch2.pt'))
            if not candidates:
                parser.error('找不到全量第2轮模型，请用 --model 指定完整路径')
            model_path = candidates[-1]
        model_path = model_path.resolve()
        if not model_path.is_file():
            parser.error('模型文件不存在')
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_sha!='4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0':
            parser.error('请选择稳定的全量第2轮模型')
        print('本次加载模型：', model_path, flush=True)
    policy = ChessCNN().to(device)
    policy.load_state_dict(torch.load(root / 'chess_model_balanced.pt', map_location=device, weights_only=True))
    policy.eval()
    evaluator = None
    if args.eval == 'neural':
        from model_residual_value import ResidualValueModel
        value_model = ResidualValueModel().to(device)
        value_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        value_model.eval()
        evaluator = TracedEvaluator(value_model)
        evaluator.validate([chess.Board()])
    engine = NeuralSearchV4(evaluator, args.seconds, args.depth, claim_draw=True)
    print(f'评价模式：{args.eval}；设备：{device}；每步搜索预算：{args.seconds} 秒')
    while True:
        color = input('你执白还是黑？输入 白 / 黑：').strip().lower()
        if color in ('白', '白方', 'w', 'white'):
            my_color = chess.WHITE
            break
        if color in ('黑', '黑方', 'b', 'black'):
            my_color = chess.BLACK
            break
        print('请输入 白 或 黑。')
    board = chess.Board()
    folder = root / 'games_v4'
    folder.mkdir(exist_ok=True)
    path = folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.pgn')

    claimed = False

    def save():
        game = chess.pgn.Game.from_board(board)
        game.headers['White'] = 'MyAI' if my_color else 'Opponent'
        game.headers['Black'] = 'Opponent' if my_color else 'MyAI'
        game.headers['EvaluationMode'] = 'neural_v4'
        game.headers['PolicyModel'] = 'chess_model_balanced.pt'
        game.headers['ValueModel'] = str(model_path) if args.eval == 'neural' else 'material'
        if args.eval == 'neural':
            game.headers['ValueModelSHA256'] = model_sha
        game.headers['CPUThreads'] = '4'
        game.headers['DrawPolicy'] = 'optional_claim'
        game.headers['SearchSeconds'] = str(args.seconds)
        game.headers['SearchDepth'] = str(args.depth)
        game.headers['Result'] = '1/2-1/2' if claimed else board.result()
        if claimed:
            game.headers['Termination'] = 'draw_claim_confirmed'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(str(game) + '\n', encoding='utf-8')
        temporary.replace(path)

    print('本地对弈或双方约定允许AI辅助的对局；请输入实际走法以保持棋盘同步。')
    print('claim：记录已成功申领的和棋。仅在实际对局接受申领后使用。')
    print('输入 SAN（Nf3、O-O）或 UCI（g1f3）；board 查看，undo 撤销一手，quit 退出。')
    print('每次记录走法后自动保存棋谱：', path)
    cached_fen, suggested, is_mate = None, None, False
    try:
        while not board.is_game_over():
            my_turn = board.turn == my_color
            if my_turn:
                if cached_fen != board.fen():
                    print('正在计算……', flush=True)
                    preferred = policy_order(board, policy, device)
                    suggested, is_mate, stats = engine.choose(board, preferred)
                    cached_fen = board.fen()
                    print(f"完成主搜索深度 {stats['depth']}；节点 {stats['nodes']:,}；搜索耗时 {stats['seconds']:.2f} 秒")
                    if stats['depth'] == 0 and not is_mate and suggested is not None:
                        print('预算内未完成第一层搜索，本次采用策略模型首选。可增加 --seconds。')
                if suggested is None:
                    print('建议申请和棋；实际申领成功后输入 claim，或输入你实际继续走的棋。')
                    text = input('请输入：').strip()
                else:
                    print(f'建议：{board.san(suggested)}（{suggested.uci()}）' + ('，一步将死' if is_mate else ''))
                    text = input('落子后回车确认，或输入你实际走的棋：').strip()
            else:
                text = input('请输入对手走法：').strip()
            if text.lower() == 'claim':
                if not board.can_claim_draw():
                    print('当前记录不满足申领条件，请核对棋谱。')
                    continue
                claimed = True
                save()
                print('已记录申领和棋。')
                break
            if text.lower() == 'quit':
                break
            if text.lower() == 'board':
                print(board)
                print(board.fen())
                continue
            if text.lower() == 'undo':
                if board.move_stack:
                    board.pop()
                    cached_fen = None
                    save()
                    print('已撤销一手记录。')
                continue
            try:
                if not text:
                    if not my_turn:
                        print('请先输入对手走法。')
                        continue
                    if suggested is None:
                        print('请明确输入 claim 或实际走法。')
                        continue
                    move = suggested
                else:
                    move = parse_move(board, text)
            except ValueError:
                print('无法识别或走法不合法，请重新输入。')
                continue
            name = board.san(move)
            board.push(move)
            cached_fen = None
            save()
            print(f'已记录：{name}（{move.uci()}）')
        if board.is_game_over():
            print('对局结束：', board.result())
    except (KeyboardInterrupt, EOFError):
        print('\n已退出。')
    finally:
        save()
        print('棋谱已保存：', path)


if __name__ == '__main__':
    main()
