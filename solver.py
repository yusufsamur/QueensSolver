class QueensPuzzleSolver:
    def __init__(self, board_size, regions, cell_regions):
        """Initialize the solver with board information"""
        self.board_size = board_size
        self.regions = regions
        self.cell_regions = cell_regions
        self.board = [[0] * board_size for _ in range(board_size)]
        
        # Sort regions by cell count (smallest first for better performance)
        self.sorted_regions = sorted(regions.items(), key=lambda x: len(x[1]))
        
        print(f"\nSolver initialized: {board_size}x{board_size} board")
        print(f"Number of regions: {len(regions)}")
        print("Regions (sorted by cell count):")
        for region_id, cells in self.sorted_regions:
            print(f"  Region {region_id}: {len(cells)} cells")
    
    def place_queen(self, pos, board_state, region_id):
        """Place a queen and mark threatened positions"""
        row, col = pos
        
        # Place queen
        board_state[row][col] = 1
        
        # Lock entire row and column
        for i in range(self.board_size):
            if board_state[row][i] == 0:
                board_state[row][i] = -1
            if board_state[i][col] == 0:
                board_state[i][col] = -1
        
        # Lock adjacent cells (Queens Game special rule)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:  # Skip the queen's position
                    continue
                    
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board_state[nr][nc] == 0:
                        board_state[nr][nc] = -1
        
        # Lock other cells in the same region
        for r, c in self.regions[region_id]:
            if board_state[r][c] == 0:
                board_state[r][c] = -1
    
    def solve_recursive(self, regions_to_place, board_state):
        """Recursive backtracking solver"""
        if not regions_to_place:  # All regions have queens
            return True
            
        current_region_id, cells = regions_to_place[0]
        remaining_regions = regions_to_place[1:]
        
        for row, col in cells:
            if board_state[row][col] == 0:  # Cell is available
                # Create a copy of the current board state
                new_state = [row[:] for row in board_state]
                self.place_queen((row, col), new_state, current_region_id)
                
                # Recursively solve for remaining regions
                if self.solve_recursive(remaining_regions, new_state):
                    # Update the original board state with the solution
                    for i in range(self.board_size):
                        for j in range(self.board_size):
                            board_state[i][j] = new_state[i][j]
                    return True
                    
        return False
    
    def solve(self):
        """Main solving method"""
        if len(self.regions) != self.board_size:
            print(f"WARNING: Region count ({len(self.regions)}) doesn't match board size!")
            if len(self.regions) > self.board_size:
                print("Too many regions, solution impossible!")
                return None
        
        # Initialize empty board (0=empty, 1=queen, -1=threatened)
        board_state = [[0] * self.board_size for _ in range(self.board_size)]
        
        if self.solve_recursive(self.sorted_regions, board_state):
            print("\nSolution found!")
            self.extract_solution(board_state)
            self.verify_solution()
            return self.board
        
        print("\nNo solution found!")
        self.analyze_failure()
        return None
    
    def extract_solution(self, board_state):
        """Extract the solution from the board state"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board[i][j] = 1 if board_state[i][j] == 1 else 0
    
    def verify_solution(self):
        """Verify that the solution meets all constraints"""
        queens = []
        valid = True
        
        # Collect queen positions
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    queens.append((i, j))
        
        print(f"Total {len(queens)} queens placed")
        
        # Check exactly one queen per region
        for region_id, cells in self.regions.items():
            count = sum(1 for r, c in cells if self.board[r][c] == 1)
            if count != 1:
                print(f"  ERROR: Region {region_id}: {count} queens")
                valid = False
        
        # Check exactly one queen per row
        for i in range(self.board_size):
            if sum(self.board[i]) != 1:
                print(f"  ERROR: Row {i}: {sum(self.board[i])} queens")
                valid = False
        
        # Check exactly one queen per column
        for j in range(self.board_size):
            col_sum = sum(self.board[i][j] for i in range(self.board_size))
            if col_sum != 1:
                print(f"  ERROR: Column {j}: {col_sum} queens")
                valid = False
        
        # Check no adjacent queens (Queens Game rule)
        for i in range(len(queens)):
            for j in range(i + 1, len(queens)):
                r1, c1 = queens[i]
                r2, c2 = queens[j]
                if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    print(f"  ERROR: Queens at ({r1},{c1}) and ({r2},{c2}) are adjacent!")
                    valid = False
        
        if valid:
            print("  ✓ All checks passed!")
        
        # Print queen positions with their regions
        print("\nQueen positions:")
        for r, c in queens:
            region_id = self.cell_regions.get((r, c), "?")
            print(f"  Queen ({r},{c}) - Region {region_id}")
    
    def analyze_failure(self):
        """Analyze why no solution was found"""
        print("\nDetailed failure analysis:")
        
        # Check for regions confined to single row/column
        problematic = []
        for region_id, cells in self.regions.items():
            rows = {r for r, c in cells}
            cols = {c for r, c in cells}
            
            if len(rows) == 1:
                problematic.append(f"Region {region_id} confined to single row (row {list(rows)[0]})")
            if len(cols) == 1:
                problematic.append(f"Region {region_id} confined to single column (column {list(cols)[0]})")
        
        if problematic:
            print("\nProblematic regions:")
            for p in problematic:
                print(f"  - {p}")
        
        # Analyze region distribution across rows and columns
        print("\nRow/column usage:")
        for i in range(self.board_size):
            row_regions = {self.cell_regions.get((i, j)) for j in range(self.board_size)}
            print(f"  Row {i}: {len(row_regions)} distinct regions")
        
        print()
        for j in range(self.board_size):
            col_regions = {self.cell_regions.get((i, j)) for i in range(self.board_size)}
            print(f"  Column {j}: {len(col_regions)} distinct regions")