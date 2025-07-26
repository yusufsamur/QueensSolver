import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import defaultdict
import threading
from sklearn.cluster import KMeans
from solver import QueensPuzzleSolver

class QueensGameSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Queens Game Solver")
        self.root.geometry("1200x960")
        
        self.board = None
        self.regions = defaultdict(list)
        self.cell_regions = {}
        self.board_size = 0
        self.solving = False
        
        self.setup_ui()
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, padx=10, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.create_title(main_container)
        self.create_control_panel(main_container)
        self.create_content_panels(main_container)
        self.create_bottom_panel(main_container)
        self.create_status_bar(main_container)
    
    def create_title(self, parent):
        title = tk.Label(parent, text="Queens Game Solver", 
                        font=('Arial', 20, 'bold'))
        title.pack(pady=10)
    
    def create_control_panel(self, parent):
        control_panel = tk.Frame(parent)
        control_panel.pack(pady=10)
        
        load_btn = tk.Button(
            control_panel, 
            text="Load From File", 
            command=self.load_from_file,
            width=20, height=2, 
            font=('Arial', 12),
            bg='#2196F3', fg='white'
        )
        load_btn.pack()
    
    def create_content_panels(self, parent):
        content_frame = tk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Original puzzle panel
        self.create_puzzle_panel(content_frame)
        
        # Solution panel
        self.create_solution_panel(content_frame)
    
    def create_puzzle_panel(self, parent):
        left_panel = tk.Frame(parent)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_panel, text="Loaded Puzzle", 
                font=('Arial', 14, 'bold')).pack()
        
        self.original_label = tk.Label(left_panel, bg='gray90', relief=tk.SUNKEN)
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_solution_panel(self, parent):
        right_panel = tk.Frame(parent)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_panel, text="Solution", 
                font=('Arial', 14, 'bold')).pack()
        
        self.solution_frame = tk.Frame(right_panel, bg='gray90', relief=tk.SUNKEN)
        self.solution_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_bottom_panel(self, parent):
        bottom_panel = tk.Frame(parent)
        bottom_panel.pack(fill=tk.X, pady=10)
        
        solve_btn = tk.Button(
            bottom_panel, 
            text="SOLVE", 
            command=self.solve_puzzle,
            width=20, height=2, 
            font=('Arial', 14, 'bold'),
            bg='#4CAF50', fg='white'
        )
        solve_btn.pack()
    
    def create_status_bar(self, parent):
        self.status_label = tk.Label(
            parent, 
            text="Load an image to begin", 
            font=('Arial', 11), 
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, pady=5)
    
    def load_from_file(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        filename = filedialog.askopenfilename(title="Select Queens puzzle image", filetypes=filetypes)
        
        if not filename:
            return
            
        image = cv2.imread(filename)
        if image is None:
            messagebox.showerror("Error", "Failed to load image!")
            return
            
        cropped = self.auto_crop_board(image)
        self.process_image(cropped if cropped is not None else image)
    
    def auto_crop_board(self, image):
        """Automatically crop the image to focus on the board area"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        aspect_ratio = w / float(h)
        if 0.9 < aspect_ratio < 1.1:  # Approximately square
            margin = int(min(w, h) * 0.02)
            x, y = max(0, x - margin), max(0, y - margin)
            w, h = min(image.shape[1] - x, w + 2 * margin), min(image.shape[0] - y, h + 2 * margin)
            return image[y:y+h, x:x+w]
        
        return None
    
    def process_image(self, image):
        """Process the loaded image and detect board structure"""
        self.status_label.config(text="Processing image...")
        self.root.update_idletasks()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        self.display_image(image_rgb)
        
        self.board_size = self.detect_board_size(image_rgb)
        if self.board_size > 0:
            self.analyze_cells(image_rgb)
        else:
            self.status_label.config(text="Board size not detected!")
    
    def display_image(self, image_rgb):
        """Display the loaded image in the GUI"""
        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(pil_image)
        
        self.original_label.config(image=photo)
        self.original_label.image = photo
    
    def detect_board_size(self, image):
        """Detect the size of the game board (n x n)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        height, width = binary.shape
        
        # Count horizontal and vertical lines to determine board size
        horizontal_counts = self.count_lines(binary, 'horizontal', height, width)
        vertical_counts = self.count_lines(binary, 'vertical', height, width)
        
        board_size = max(max(horizontal_counts), max(vertical_counts))
        print(f"Detected board size: {board_size}")
        return board_size
    
    def count_lines(self, binary, direction, height, width):
        """Count grid lines in the given direction"""
        counts = []
        threshold_run = 20  # Minimum cell width/height
        
        if direction == 'horizontal':
            for _ in range(5):
                y = np.random.randint(height // 4, 3 * height // 4)
                counts.append(self.count_line(binary[y, :], threshold_run))
        else:
            for _ in range(5):
                x = np.random.randint(width // 4, 3 * width // 4)
                counts.append(self.count_line(binary[:, x], threshold_run))
        
        return counts
    
    def count_line(self, line, threshold):
        """Count continuous segments in a line"""
        count = 0
        run_length = 0
        
        for pixel in line:
            if pixel > 128:  # White pixel (cell area)
                run_length += 1
            else:  # Black pixel (grid line)
                if run_length >= threshold:
                    count += 1
                run_length = 0
        
        if run_length >= threshold:  # Last segment
            count += 1
            
        return count
    
    def analyze_cells(self, image):
        """Analyze cells and detect regions"""
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.regions = defaultdict(list)
        self.cell_regions = {}
        
        height, width = image.shape[:2]
        cell_height, cell_width = height / self.board_size, width / self.board_size
        
        cell_colors = self.extract_cell_colors(image, height, width, cell_height, cell_width)
        self.assign_regions(cell_colors)
        self.create_solution_board()
    
    def extract_cell_colors(self, image, height, width, cell_height, cell_width):
        """Extract average colors for each cell"""
        cell_colors = []
        margin_factor = 0.2  # Percentage of cell size to ignore edges
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                # Get center coordinates of the cell
                cy, cx = int((i + 0.5) * cell_height), int((j + 0.5) * cell_width)
                margin = int(min(cell_height, cell_width) * margin_factor)
                
                # Define region around center
                y1, y2 = max(0, cy - margin), min(height, cy + margin)
                x1, x2 = max(0, cx - margin), min(width, cx + margin)
                
                cell_region = image[y1:y2, x1:x2]
                if cell_region.size == 0:
                    avg_color = (0, 0, 0)  # Default to black if empty
                else:
                    # Calculate average color and round to reduce variations
                    avg_color = tuple((np.mean(cell_region.reshape(-1, 3), axis=0) // 20 * 20).astype(int))
                
                cell_colors.append(avg_color)
        
        return cell_colors
    
    def assign_regions(self, cell_colors):
        """Assign cells to regions based on their colors"""
        unique_colors = list(set(cell_colors))
        color_dict = {color: idx for idx, color in enumerate(unique_colors)}
        
        if len(unique_colors) > self.board_size:
            self.cluster_colors(cell_colors)
        else:
            self.assign_by_color(cell_colors, color_dict)
        
        # Ensure we have exactly board_size regions
        if len(self.regions) != self.board_size:
            self.reassign_with_kmeans(cell_colors)
        
        print(f"\nRegion analysis (Total {len(self.regions)} regions):")
        for region_id, cells in self.regions.items():
            print(f"Region {region_id}: {len(cells)} cells")
        
        self.status_label.config(text=f"Detection complete. Found {len(self.regions)} regions.")
    
    def cluster_colors(self, cell_colors):
        """Use K-means clustering when too many colors are detected"""
        unique_colors = np.array(list(set(cell_colors)))
        kmeans = KMeans(n_clusters=self.board_size, random_state=42, n_init=10)
        kmeans.fit(unique_colors)
        
        color_to_label = {tuple(color): label for color, label in zip(unique_colors, kmeans.labels_)}
        
        idx = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                label = color_to_label[cell_colors[idx]]
                self.cell_regions[(i, j)] = label
                self.regions[label].append((i, j))
                idx += 1
    
    def assign_by_color(self, cell_colors, color_dict):
        """Directly assign regions when color count matches board size"""
        idx = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                color = cell_colors[idx]
                label = color_dict[color]
                self.cell_regions[(i, j)] = label
                self.regions[label].append((i, j))
                idx += 1
    
    def reassign_with_kmeans(self, cell_colors):
        """Force region count to match board size using K-means"""
        all_colors = np.array(cell_colors)
        kmeans = KMeans(n_clusters=self.board_size, random_state=42, n_init=20)
        labels = kmeans.fit_predict(all_colors)
        
        self.regions = defaultdict(list)
        self.cell_regions = {}
        
        idx = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                label = labels[idx]
                self.cell_regions[(i, j)] = label
                self.regions[label].append((i, j))
                idx += 1
    
    def create_solution_board(self):
        """Create the visual representation of the solution board"""
        for widget in self.solution_frame.winfo_children():
            widget.destroy()
            
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#C7CEEA',
                  '#FFA07A', '#98D8C8', '#F8B195', '#B4A7D6', '#87CEEB']
        
        self.solution_cells = []
        board_container = tk.Frame(self.solution_frame)
        board_container.pack(expand=True)
        
        for i in range(self.board_size):
            row_cells = []
            for j in range(self.board_size):
                region_id = self.cell_regions.get((i, j), -1)
                color = colors[region_id % len(colors)] if region_id >= 0 else 'white'
                
                cell = tk.Label(
                    board_container, 
                    text='', 
                    width=5, 
                    height=2,
                    font=('Arial', 16), 
                    bg=color, 
                    relief=tk.RAISED
                )
                cell.grid(row=i, column=j, padx=1, pady=1)
                row_cells.append(cell)
            self.solution_cells.append(row_cells)
    
    def solve_puzzle(self):
        """Initiate the puzzle solving process"""
        if not self.board_size or not self.regions:
            messagebox.showwarning("Warning", "Please load a puzzle first!")
            return
            
        self.solving = True
        self.status_label.config(text="Searching for solution...")
        
        threading.Thread(
            target=self._solve_puzzle, 
            daemon=True
        ).start()
    
    def _solve_puzzle(self):
        """Solve the puzzle in a separate thread"""
        solver = QueensPuzzleSolver(self.board_size, self.regions, self.cell_regions)
        solution = solver.solve()
        
        if solution:
            self.display_solution(solution)
            self.root.after(0, lambda: self.status_label.config(text="Solution found!"))
        else:
            self.root.after(0, lambda: self.status_label.config(text="No solution found!"))
            
        self.solving = False
    
    def display_solution(self, solution):
        """Display the solution on the board"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if solution[i][j] == 1:
                    self.solution_cells[i][j].config(text='♛', font=('Arial', 24))
                else:
                    self.solution_cells[i][j].config(text='')