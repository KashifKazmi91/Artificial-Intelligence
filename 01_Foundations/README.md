# Level 1: Foundations

## 📖 Overview

This module introduces the **core concepts of Artificial Intelligence** and establishes the foundation for all advanced topics. You'll learn how to represent problems, understand different search strategies, and implement algorithms to find solutions.

**Focus**: Classical AI approaches, problem-solving, and systematic search methods.

---

## 🎯 What You'll Learn

By completing this level, you will understand:

✅ What Artificial Intelligence is and its goals  
✅ How to represent problems in a way computers can solve  
✅ Basic search algorithms (BFS, DFS, Uniform Cost Search)  
✅ Heuristic-based search (Greedy, A* algorithm)  
✅ How to measure algorithm performance  
✅ Trade-offs between different search strategies  

---

## 📚 Modules in This Level

### **Module 01: Introduction to AI**
**⏱️ Duration**: 2-3 hours | **Difficulty**: Beginner

**What You'll Learn**:
- Definition of Artificial Intelligence
- Goals and applications of AI
- Problem-solving approach in AI
- State space representation
- Goal formulation and constraints

**Hands-On Exercises**:
- Define a real-world problem as an AI problem
- Identify initial state, goal state, and actions
- Compare different problem formulations
- Discuss AI applications in different domains

**By the End**: You'll understand what AI does and how to think about problems in AI terms.

---

### **Module 02: Problem Representation & State Space**
**⏱️ Duration**: 3-4 hours | **Difficulty**: Beginner

**What You'll Learn**:
- How to represent problems formally
- State space definition and structure
- Initial state, goal state, intermediate states
- Actions and state transitions
- Graph representation of problems
- Problem abstraction and refinement

**Hands-On Exercises**:
- Represent classic puzzles (8-puzzle, Tower of Hanoi)
- Design state spaces for real problems
- Implement state transition functions
- Visualize state graphs
- Analyze problem complexity

**By the End**: You'll be able to take any problem and represent it as a state space that an AI algorithm can solve.

---

### **Module 03: Search Strategies Part 1 (Uninformed Search)**
**⏱️ Duration**: 4-5 hours | **Difficulty**: Intermediate

**What You'll Learn**:
- Breadth-First Search (BFS): explores level by level
- Depth-First Search (DFS): explores deeply before backtracking
- Uniform Cost Search (UCS): explores based on path cost
- Completeness: Does algorithm always find solution?
- Optimality: Does algorithm find best solution?
- Time and space complexity analysis

**Hands-On Exercises**:
- Implement BFS from scratch
- Implement DFS from scratch
- Implement Uniform Cost Search
- Compare performance on different problems
- Analyze memory usage
- Trace algorithm execution step-by-step
- Solve actual puzzle problems

**By the End**: You'll understand why different searches work differently and when to use each one.

---

### **Module 04: Search Strategies Part 2 (Informed Search)**
**⏱️ Duration**: 4-5 hours | **Difficulty**: Intermediate-Advanced

**What You'll Learn**:
- Heuristic functions: Estimating distance to goal
- Greedy Best-First Search: Always go toward goal
- A* Algorithm: Combine cost and heuristic
- Heuristic admissibility: Guarantees optimality
- Heuristic consistency: Ensures efficiency
- A* optimality and completeness proofs

**Hands-On Exercises**:
- Design heuristic functions for different problems
- Implement Greedy Best-First Search
- Implement A* Algorithm
- Compare A* vs uninformed search
- Analyze heuristic quality
- Understand importance of good heuristics
- Solve complex puzzles efficiently

**By the End**: You'll know how to guide search with heuristics and make algorithms thousands of times faster.

---

## 🔄 Recommended Learning Path

```
START HERE
    ↓
📖 Read: What is AI? (Concept Overview)
    ↓
🔧 Try: Understand problem formulation
    ↓
💻 Code: Implement a simple state space
    ↓
▶️ Run: Visualize state transitions
    ↓
📋 Exercise: Represent your first problem
    ↓
⬇️ MOVE TO MODULE 02
    ↓
📖 Read: State space representation
    ↓
🔧 Try: Build state spaces for 3 problems
    ↓
💻 Code: Implement transitions
    ↓
▶️ Run: Visualize your state graphs
    ↓
⬇️ MOVE TO MODULE 03
    ↓
📖 Read: How search algorithms work
    ↓
🔧 Try: Trace BFS/DFS by hand
    ↓
💻 Code: Implement BFS, DFS, UCS
    ↓
▶️ Run: Solve 8-puzzle problem
    ↓
📊 Compare: Time, space, solution quality
    ↓
⬇️ MOVE TO MODULE 04
    ↓
📖 Read: How heuristics guide search
    ↓
🔧 Try: Design heuristics
    ↓
💻 Code: Implement A* algorithm
    ↓
▶️ Run: Compare A* vs uninformed search
    ↓
✅ READY FOR LEVEL 2!
```

---

## 💡 Key Concepts Explained

### **State Space**
A complete description of:
- All possible configurations in a problem
- How to transition between configurations
- Which states are valid/invalid

**Example**: In 8-Puzzle, state space includes all 362,880 possible tile arrangements.

### **Problem Formulation**
The process of deciding:
- What is the initial state?
- What is the goal state?
- What actions are possible?
- What are the constraints?

### **Search Algorithm**
A systematic process to:
- Explore states in the state space
- Find path from initial to goal state
- Return the solution or report failure

### **Heuristic Function**
An educated estimate that:
- Predicts distance from any state to goal
- Guides search toward solution
- Must not overestimate (admissible heuristic)

---

## 🎓 Learning Tips

### 📝 **Before Each Module**:
1. Read the concept explanations carefully
2. Study the theory sections
3. Look at example problems
4. Understand the algorithm's logic

### 💻 **During Code**:
1. Don't copy-paste—type the code yourself
2. Read every comment line
3. Run code cells one at a time
4. Modify code and see what changes
5. Add print statements to debug

### 🔍 **After Each Module**:
1. Trace an algorithm by hand
2. Solve a problem manually
3. Compare with algorithm output
4. Answer practice questions
5. Explain to someone else

### 🚀 **Challenge Yourself**:
1. Solve increasingly complex problems
2. Design better heuristic functions
3. Optimize code for efficiency
4. Combine concepts in new ways

---

## 📊 Comparison of Search Algorithms

| Aspect | BFS | DFS | UCS | Greedy | A* |
|--------|-----|-----|-----|--------|-----|
| **Explores** | Breadth | Depth | By cost | Toward goal | Cost + goal |
| **Optimal** | ✅ Yes | ❌ No | ✅ Yes* | ❌ No | ✅ Yes* |
| **Complete** | ✅ Yes | ❌ Maybe | ✅ Yes | ❌ Maybe | ✅ Yes |
| **Memory** | High | Low | High | Low | Medium |
| **Speed** | Medium | Fast | Slow | Fast | Very Fast |
| **When to use** | Small graphs | Limited memory | Weighted paths | With good heuristic | Best for most |

*When heuristic is admissible

---

## 🛠️ Tools & Libraries You'll Use

```python
# Data structures
from collections import deque, heapq  # For search queues
import networkx as nx                 # For graph operations

# Visualization
import matplotlib.pyplot as plt        # Plot graphs
import matplotlib.patches as patches  # Draw shapes

# Utilities
import numpy as np                     # Array operations
from typing import List, Dict, Tuple  # Type hints
```

---

## 📖 Glossary

**AI**: Creating intelligent machines that can perceive, reason, and act autonomously.

**State**: A complete description of a configuration in a problem.

**State Space**: All possible states and transitions between them.

**Action**: A move or transition from one state to another.

**Goal Test**: Function checking if current state is goal.

**Path Cost**: Total cost of a sequence of actions.

**Search Algorithm**: Method to explore state space finding solution.

**Completeness**: Algorithm always finds solution if one exists.

**Optimality**: Algorithm finds best (lowest cost) solution.

**Heuristic**: Educated guess estimating remaining distance to goal.

**Admissible Heuristic**: Never overestimates distance to goal.

---

## 📚 Recommended Reading

**Primary**:
- Chapter 3 "Solving Problems by Searching" - AI: A Modern Approach
- Chapter 4 "Informed Search Strategies"

**Supplementary**:
- Wikipedia articles on BFS, DFS, A*
- Academic papers on heuristic search
- Algorithm visualization websites

---

## ❓ Frequently Asked Questions

**Q: Why do we need state space representation?**  
A: It lets us model ANY problem in a uniform way that algorithms can solve.

**Q: Which search algorithm should I use?**  
A: BFS for unweighted graphs, A* with good heuristic for most problems.

**Q: What makes a good heuristic?**  
A: Fast to compute, never overestimates, gives useful guidance.

**Q: Can I use A* without a heuristic?**  
A: Yes, but then it behaves like UCS—slow and memory-intensive.

**Q: How do I know if my heuristic is admissible?**  
A: It must never overestimate actual distance to goal state.

---

## ✅ Completion Checklist

- [ ] Completed Module 01 notebook
- [ ] Understood problem formulation
- [ ] Completed Module 02 notebook
- [ ] Can represent problems as state spaces
- [ ] Completed Module 03 notebook
- [ ] Implemented BFS, DFS, UCS
- [ ] Completed Module 04 notebook
- [ ] Implemented Greedy and A*
- [ ] Compared all search algorithms
- [ ] Answered all practice questions
- [ ] Can explain each algorithm to someone else
- [ ] Solved sample problems with each algorithm
- [ ] **Ready for Level 2!** ✅

---

## 🔗 Links to Modules

- [📓 Module 01: Introduction to AI](./01_Introduction_to_AI.ipynb)
- [📓 Module 02: Problem Representation](./02_Problem_Representation.ipynb)
- [📓 Module 03: Search Strategies Part 1](./03_Search_Strategies_Part1.ipynb)
- [📓 Module 04: Search Strategies Part 2](./04_Search_Strategies_Part2.ipynb)

---

## 💬 Need Help?

1. **Re-read** the explanation section
2. **Study** the inline code comments
3. **Trace** algorithm execution by hand
4. **Modify** code and experiment
5. **Ask** questions in repository issues

---

**Ready? Start with [Module 01 →](./01_Introduction_to_AI.ipynb)**

**Remember**: Understanding each concept deeply is more valuable than completing quickly. 🚀
