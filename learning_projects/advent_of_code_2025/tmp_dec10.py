from collections import deque

def min_presses_bfs(buttons: list[tuple[int, ...]], targets: list[int]) -> int:
    """
    Find minimum button presses using BFS with pruning.
    """
    num_counters = len(targets)
    initial_state = tuple([0] * num_counters)
    target_state = tuple(targets)
    
    if initial_state == target_state:
        return 0
    
    queue = deque([(initial_state, 0)])  # (state, presses)
    visited = {initial_state: 0}
    
    while queue:
        state, presses = queue.popleft()
        
        # Try pressing each button
        for button in buttons:
            new_state = list(state)
            
            # Apply button press
            for counter_idx in button:
                new_state[counter_idx] += 1
            
            new_state = tuple(new_state)
            
            # Prune: if any counter exceeds target, skip
            if any(new_state[i] > targets[i] for i in range(num_counters)):
                continue
            
            # Found solution
            if new_state == target_state:
                return presses + 1
            
            # Avoid revisiting states with more presses
            if new_state not in visited or visited[new_state] > presses + 1:
                visited[new_state] = presses + 1
                queue.append((new_state, presses + 1))
    
    return None  # No solution found


# Test with your example
buttons = [(0,1,2,3,7,8,9), (0,1,2,5,6,9), (0,1,2,3,5,6,8,9), 
           (1,2,4,8,9), (0,1,2,3,5,6,9), (2,3,4,5,6,7,8,9), 
           (1,8), (0,1,2,4,6,8,9), (3,5,7,8), (1,2,6,9)]
targets = [60,81,198,179,144,185,175,155,202,198]

result = min_presses_bfs(buttons, targets)
print(f"BFS result: {result}")