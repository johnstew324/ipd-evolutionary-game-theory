import numpy as np

class IPDStrategy:
    def __init__(self, name):
        self.name = name
        self.last_opponent_move = None

    def make_move(self, opponent_history=None):
        pass
    
    def reset(self):
        self.last_opponent_move = None


class AlwaysCooperate(IPDStrategy):
    def __init__(self):
        super().__init__("Always Cooperate")
    
    def make_move(self, opponent_history=None):
        return 0  


class AlwaysDefect(IPDStrategy):
    def __init__(self):
        super().__init__("Always Defect")
    
    def make_move(self, opponent_history=None):
        return 1  


class TitForTat(IPDStrategy):
    def __init__(self):
        super().__init__("Tit for Tat")
    
    def make_move(self, opponent_history=None):
        if opponent_history is None or len(opponent_history) == 0:
            return 0
        return opponent_history[-1]


class RandomStrategy(IPDStrategy):
    def __init__(self):
        super().__init__("Random")
    
    def make_move(self, opponent_history=None):
        return np.random.choice([0, 1])


class SuspiciousTitForTat(IPDStrategy):
    def __init__(self):
        super().__init__("Suspicious Tit for Tat")
    
    def make_move(self, opponent_history=None):
        if opponent_history is None or len(opponent_history) == 0:
            return 1
        return opponent_history[-1]


class GrudgerStrategy(IPDStrategy):
    def __init__(self):
        super().__init__("Grudger")
        self.defected = False
    
    def make_move(self, opponent_history=None):
        if opponent_history is None or len(opponent_history) == 0:
            return 0
        if 1 in opponent_history:
            self.defected = True
        if self.defected:
            return 1
        return 0
    
    def reset(self):
        super().reset()
        self.defected = False


class TitForTwoTats(IPDStrategy):
    def __init__(self):
        super().__init__("Tit for Two Tats")
    
    def make_move(self, opponent_history=None):
        if opponent_history is None or len(opponent_history) < 2:
            return 0 
        
        if opponent_history[-1] == 1 and opponent_history[-2] == 1:
            return 1
        return 0 


class GeneticStrategy(IPDStrategy):
    def __init__(self, genome, memory_length=1):
        super().__init__("Genetic")
        self.genome = genome
        self.memory_length = memory_length
        self.history = [] 
    
    def make_move(self, opponent_history=None):
        if opponent_history is None:
            opponent_history = []
    
        if len(opponent_history) == 0:
            move = self.genome[0]
            self.history.append(move)
            return move

        if self.memory_length == 1:
            idx = 1 + opponent_history[-1]
            move = self.genome[idx]
            self.history.append(move)
            return move
        elif self.memory_length == 2:
            if len(opponent_history) == 1:
                idx = 1 + opponent_history[-1]
                move = self.genome[idx]
            else:
                key = (opponent_history[-2] << 1) + opponent_history[-1]
                idx = 1 + key
                move = self.genome[idx]
            
            self.history.append(move)
            return move
        
        elif self.memory_length == 3:
            if len(opponent_history) == 0:
                move = self.genome[0]
            elif len(self.history) == 0 or len(opponent_history) == 0:
                move = self.genome[1]
            else:
                key = (self.history[-1] << 1) + opponent_history[-1]
                idx = 1 + key
                move = self.genome[idx]
            
            self.history.append(move)
            return move
    
    def reset(self):
        super().reset()
        self.history = []


class ProbabilisticGeneticStrategy(IPDStrategy):
    def __init__(self, genome, memory_length=1):
        super().__init__("Probabilistic Genetic")
        self.genome = genome 
        self.memory_length = memory_length
        self.history = []
    
    def make_move(self, opponent_history=None):
        if opponent_history is None:
            opponent_history = []
        
        if len(opponent_history) == 0:
            if np.random.random() < self.genome[0]:
                move = 0  
            else:
                move = 1 
            self.history.append(move)
            return move
        
        if self.memory_length == 1:
            idx = 1 + opponent_history[-1]
            if np.random.random() < self.genome[idx]:
                move = 0  
            else:
                move = 1
            self.history.append(move)
            return move
        
    def reset(self):
        super().reset()
        self.history = []