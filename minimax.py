import random

def minimax_move(board, level):
    """
    Realiza a jogada do Minimax com base no nível de dificuldade.
    - `board`: Estado atual do tabuleiro (matriz 3x3).
    - `level`: Nível de dificuldade ("easy", "medium", "hard").
    """
    if level == "easy":
        if random.random() < 0.75:
            return random_move(board)
        else:
            return best_move(board)
    elif level == "medium":
        if random.random() < 0.5:
            return random_move(board)
        else:
            return best_move(board)
    elif level == "hard":
        return best_move(board)

def random_move(board):
    """
    Escolhe uma jogada aleatória válida.
    """
    empty_cells = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ""]
    return random.choice(empty_cells)

def best_move(board):
    """
    Calcula a melhor jogada usando o algoritmo Minimax.
    """
    best_score = -float("inf")
    move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == "":
                board[i][j] = "O"
                score = minimax(board, 0, False)
                board[i][j] = ""
                if score > best_score:
                    best_score = score
                    move = (i, j)

    return move

def minimax(board, depth, is_maximizing):
    """
    Algoritmo recursivo do Minimax.
    - `board`: Estado atual do tabuleiro.
    - `depth`: Profundidade da recursão.
    - `is_maximizing`: Se é a vez do jogador Max (rede neural).
    """
    winner = check_winner(board)
    if winner == "network":
        return -1
    elif winner == "minimax":
        return 1
    elif is_draw(board):
        return 0

    if is_maximizing:
        best_score = -float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "O"
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ""
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "X"
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ""
                    best_score = min(score, best_score)
        return best_score

def check_winner(board):
    """
    Verifica o vencedor no tabuleiro.
    """
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "":
            return "network" if board[i][0] == "X" else "minimax"
        if board[0][i] == board[1][i] == board[2][i] != "":
            return "network" if board[0][i] == "X" else "minimax"
    if board[0][0] == board[1][1] == board[2][2] != "" or board[0][2] == board[1][1] == board[2][0] != "":
        return "network" if board[1][1] == "X" else "minimax"
    return None

def is_draw(board):
    """
    Verifica se o jogo empatou.
    """
    return all(cell != "" for row in board for cell in row)
