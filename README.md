# Evolutionary Game Theory:  Iterated Prisoner’s Dilemma (IPD)

## Overview
This project applies **Genetic Algorithms (GAs)** to evolve effective strategies for the **Iterated Prisoner’s Dilemma (IPD)**, a foundational problem in game theory.  
It builds on previously built GA used for the Travelling Salesman Problem (TSP), adapting the same evolutionary mechanisms (tournament selection, elitism, mutation, and crossover) to a game-theory setting.

The project is divided into **two parts**:
- **Part 1: Strategy Evolution:**  
  Evolving binary, memory-based, and probabilistic IPD strategies against fixed opponents.  
- **Part 2: Noise and Robustness Analysis:**  
  Investigating how communication errors (“noise”) affect the evolution and robustness of these strategies.

---

## The Iterated Prisoner’s Dilemma

Two players repeatedly choose between:
- **Cooperate (C)**
- **Defect (D)**

|             | Opponent C | Opponent D |
|--------------|-------------|-------------|
| **Player C** | (3, 3)      | (0, 5)      |
| **Player D** | (5, 0)      | (1, 1)      |

This creates a tension between **short-term, self-interest** and **long-term cooperation**.  
The overall challenge is to evolve strategies that maximise total payoff across many rounds and opponents.

---

## Implementation Summary

### Genetic Algorithm Setup

| Parameter | Value / Description |
|------------|--------------------|
| Population size | 100 |
| Elitism | 10 best individuals |
| Tournament size | 5 |
| Crossover rate | 0.7 |
| Mutation rate | 0.1 |
| Generations | 100 |
| Rounds per game | 200 |

Fitness = Σ (score against each opponent)


### Strategy Representations

| Type | Description | Example |
|------|--------------|---------|
| **Memory-1 (Binary)** | Considers the last move of the opponent | `[0, 1, 0] → Cooperate, Defect after C, Cooperate after D` |
| **Memory-2 (Binary)** | Considers last two moves of the opponent | `[1, 0, 1, 1, 0] → Defect, Cooperate after CC, Defect after CD/DC, Cooperate after DD` |
| **Probabilistic** | Uses probabilities (0–1) for cooperation | `[0.8, 0.3, 0.6] → 80% coop first move, 30% after C, 60% after D` |


### Opponent Pool

To ensure a diverse environment:

- Always Cooperate (All-C)  
- Always Defect (All-D)  
- Tit-for-Tat (TFT)  
- Suspicious TFT  
- Grudger (Grim Trigger)  
- Tit-for-Two-Tats  
- Random Strategy  

---

## Part 1: Strategy Evolution

### Memory-1
- Evolved strategy identical to **Tit-for-Tat**  
- Achieved fitness ≈ **3552**  
- High performance vs cooperative opponents, poor vs All-Defect  

### Memory-2
- Best strategy: `[C, C, C, D, D]`  
- Slight improvement (fitness ≈ **3607**)  
- Better recovery from negative cycles (e.g., vs Suspicious TFT)  

### Probabilistic
- Best strategy: cooperate except 5% chance of forgiveness after defection  
- Highest fitness ≈ **3658**  
- Small forgiveness improves adaptability and stability  

---

## Part 2: Noise & Robustness

To model **real-world uncertainty**, a noise level (0%, 5%, 10%, 20%) randomly swtichs a player’s intended move (C ↔ D).

### Observations
- Increasing noise drives evolution toward **defection** strategies.  
- Where as in noise-free environments **Tit-for-Tat** dominates.  
- At 5–10% noise **Always Defect** is strongest.  
- 20% noise the most adapatable stragie is **Initially cooperate, then defect**.

### Robustness Findings
- Strategies evolved in noisy environments are **more adaptable overall**.  
- Noise-free strategies perform poorly when tested under noise.  
- The 5% noise-trained strategy performed best across multiple noise conditions.  

---

## Example Outputs

All experiment plots (fitness progressions, cooperation analysis, robustness curves) are saved to:

results/
├── fitness_mem1.png
├── fitness_mem2.png
├── probabilistic_fitness.png
├── noise_fitness_history.png
├── noise_robustness.png
├── cooperation_rate_analysis.png
└── interaction_types_analysis.png

View Notebook for futher analysis


---

## Run Programme
1. Clone the repository:
```bash
git clone https://github.com/johnstew324/ipd-evolutionary-game-theory.git
cd ipd-evolutionary-game-theory/src
```
2. Install any required depedencies ( python, numpy, matplotlib)
```bash
pip install -r requirements.txt
```
3. Run experiments 
```bash
python main.py
```
4. All plots/results saved to directory ```../plots```

---

## Key Learnings
* Genetic algorithms successfully discover the most effective IPD strategies.
* Memory depth and forgiveness enhance cooperation.
* Noise introduces realism, driving strategies toward self-protection and adaptability.
* Robust strategies evolve under uncertainty, not perfection.

--- 

## References

* [Stanford Encyclopedia of Philosophy - Prisoner’s Dilemma: Strategies for the Iterated Prisoner’s Dilemma ](https://plato.stanford.edu/entries/prisoner-dilemma/strategy-table.html)  
* [Chong S.Y (2007)](https://www.worldscientific.com/doi/abs/10.1142/9789812770684_0002) 