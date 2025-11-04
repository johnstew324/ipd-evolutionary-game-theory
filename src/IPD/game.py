from .strategies import IPDStrategy

class IPDGame:
    def __init__(self, rounds: int = 200):
        self.rounds = rounds
        
        # Payoff matrix: [my_move][opponent_move] -> (my_score, opponent_score)
        # C = 0, D = 1
        # (C,C) = (3,3), (C,D) = (0,5), (D,C) = (5,0), (D,D) = (1,1)
        
        self.payoff_matrix = [
            [(3, 3), (0, 5)],  
            [(5, 0), (1, 1)]   
        ]
    
    
    
    def play_round(self, player1_move, player2_move):
        return self.payoff_matrix[player1_move][player2_move]
    
    def play_game(self, player1, player2):
        player1.reset()
        player2.reset()
        
        player1_history = []
        player2_history = []
        player1_score = 0
        player2_score = 0
        
        for _ in range(self.rounds):
            player1_move = player1.make_move(player2_history)
            player2_move = player2.make_move(player1_history)
            
            player1_history.append(player1_move)
            player2_history.append(player2_move)
            
            round_score = self.play_round(player1_move, player2_move)
            player1_score += round_score[0]
            player2_score += round_score[1]
        
        return player1_score, player2_score
    
    def play_tournament(self, player, opponents):
        results = {}
        for opponent in opponents:
            player_score, _ = self.play_game(player, opponent)
            results[opponent.name] = player_score
        
        return results